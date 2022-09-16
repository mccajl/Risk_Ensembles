#!/usr/bin/env python3

"""
Usage:

Example cartpole command (~8k ts to solve):
    python main.py \
        --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
        --replay_size 5000 --starts_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
        --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
"""
import argparse
import math
import random
from copy import deepcopy
import os
import numpy as np
import torch
import torch.optim as optim
from DQN.helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from DQN.models import DQN, CnnDQN, DropoutDQN
from risk_estimation.risk_estimation import Approximator, Trainer, Approx_Buffer, Rollout
from best_frozen_ensembles.test_ensemble import roll_ensemble, roll_single_agent
from best_frozen_ensembles.train_avfs import train_avf
import matplotlib.pyplot as plt
from PolicyEnsembles.agent import Agent
from PolicyEnsembles.agent import DQN_Trainer
from PolicyEnsembles.ensemble import Ensemble
from itertools import combinations


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor




def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def generate_combinations(cur_ensemble, candidate_agents, n):
    # Generate all possible ensembles from the current ensemble and the candidate agents
    all_agents = list(cur_ensemble) + list(candidate_agents)
    possible_ensembles = combinations(all_agents, n)
    return possible_ensembles


def get_ens_variance(agents, avf_trainers, env):
    # Calculate the variance among the average state feature each agent encounters. Take the mean of those variances.
    ens_state_vals = []

    for i in range(len(agents)):
        if agents[i].state_mean is None:
            roll = Rollout(agents[i], deepcopy(env))
            agents[i].calculate_state_distribution(avf_trainers[agents[i].agent_id], roll) # Get risk estimator trainer corresponding to agent id
        state_mean, state_var = agents[i].get_state_distribution()
        ens_state_vals.append(state_mean)

    ens_feat_var = np.var(np.array(ens_state_vals), axis=0)
    avg_state_var = np.mean(ens_feat_var)
    return avg_state_var

def form_objective(agents, avf_trainers, env):
    state_var = get_ens_variance(agents, avf_trainers, env)
    objective = state_var
    return objective

def choose_ensemble(current_ensemble, possible_ensembles, avf_trainers, env):
    best_objective = form_objective(current_ensemble, avf_trainers, env)
    ens_agents = deepcopy(current_ensemble)
    replaced = False
    for i, ens in enumerate(possible_ensembles):
        ens = list(ens)
        test_objective = form_objective(ens, avf_trainers, env)
        #print(test_objective)
        if test_objective > best_objective:
            best_objective = test_objective
            ens_agents = deepcopy(ens)
            replaced = True
    
    return ens_agents, replaced, best_objective
    

def run_gym_ensemble(params):
    if params.CnnDQN:
        env = make_atari(params.env)
        env = wrap_pytorch(wrap_deepmind(env))
        q_network = CnnDQN(env.observation_space.shape, env.action_space.n)
        target_q_network = deepcopy(q_network)
    else:
        env = make_gym_env(params.env)
        q_network = DQN(env.observation_space.shape, env.action_space.n)
        target_q_network = deepcopy(q_network)

    if USE_CUDA:
        q_network = q_network.cuda()
        target_q_network = target_q_network.cuda()


    avf_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]])), Approx_Buffer())

            
    env.seed(params.seed)
    agent = Agent(env, q_network, target_q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    replay_buffer = ReplayBuffer(params.replay_size)

    dqn_trainer = DQN_Trainer(agent, optimizer, replay_buffer, params)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()

    if params.save_transition:
        if not os.path.exists(params.transition_dir):
            os.makedirs(params.transition_dir, exist_ok=True)

    save_transitions = False
    num_training_iterations = 0
    states = []
    episode_failures = []

    ensemble_rewards = {}
    single_agent_rewards = {}

    saved_agents = []
    avf_trainers = {} #We use a separate risk estimation trainer for every checkpointed agent. Dictionary to hash by episodes trained

    K = params.K # Every K checkpoints, choose an ensemble based on state diversity
    ensemble_n = params.ensemble_n # Size of ensemble
    T = params.save_every # Save a new agent every T episodes. New agent is a candidate for the next ensemble choosing
    avf_trainers[T] = avf_trainer
    current_agent_id = T
    prev_agent_id = None

    episodes_completed = 0


    candidate_ensembles = [] # We will fill this with a new possible ensemble every K * T episodes
    candidate_agents = [] # This contains the current candidates for the next ensemble. Emptied every K * T episodes
    current_ensemble = [] # Current 'best' (according to state diversity) ensemble. Starts empty
    saved = False
    for ts in range(1, params.max_ts + 1):
        
        if episodes_completed % T == 0 and episodes_completed > 0 and not saved:
            saved = True
            agent.save_weights(params.save_dir, episodes_completed) #Save with episode iteration (for testing with AVF)
            checkpoint_agent = Agent(env, deepcopy(agent.q_network), deepcopy(agent.target_q_network), agent_id=episodes_completed)
            saved_agents.append(deepcopy(checkpoint_agent))
            next_avf_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]])), Approx_Buffer())
            avf_trainers[episodes_completed + T] = next_avf_trainer # Initialize avf trainer for the agent at the next checkpoint
            prev_agent_id = current_agent_id
            current_agent_id = episodes_completed + T

            candidate_agents.append(deepcopy(checkpoint_agent))
            if len(current_ensemble) < ensemble_n: #Should only trigger at the start of training
                current_ensemble.append(deepcopy(checkpoint_agent))
            
            """
            Small temporary bug:
            the last candidate agent will have T less transitions in its avf buffer for
            the calculation of its state distribution - since the next agent has not run its
            timesteps yet. This just means the state dist of the last candidate agent will be
            less accurate. Could try to fix later but it makes the code messy
            """
 
            if len(candidate_agents) == K: # Choose a new ensemble out of the current ensemble and the candidate agents
                possible_ensembles = generate_combinations(current_ensemble, candidate_agents, ensemble_n)
                current_ensemble, changed, obj_value = choose_ensemble(current_ensemble, possible_ensembles,
                                                                avf_trainers, deepcopy(env))
                if changed:
                    print("New ensemble chosen. {} state variance.".format(obj_value))
                    candidate_ensembles.append(current_ensemble)
                
                candidate_agents = [] #Reset new batch of candidates


        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )
        action, value = agent.act(state, epsilon)
        

        next_state, reward, done, info = env.step(int(action.cpu()))

        states.append(state)
        assert 'failure' in info
        episode_failures.append(info['failure'])

        dqn_trainer.push_replay(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        
        if done:
            if saved:
                saved = False
            """
            Pad sequences with failure (1 or 0) based on if agent fails within H timesteps
            """
            fail_padding = pad_failures(episode_failures, params.H)
            #Note - since we add experience in this way, we are using the data between the last saved agent and the next saved agent
            #to train the current agent's risk estimator - this is less accurate than using frozen policy data
            avf_trainers[current_agent_id].add_experience(np.array(states), fail_padding) #Add episode experience to the latest checkpointed agent
            #if prev_agent_id is not None: #Add experience to both current and previous agent (experience of an agent is centered around its checkpoint)
                #avf_trainers[prev_agent_id].add_experience(np.array(states), fail_padding)

            states = []
            episode_failures = []

            episodes_completed += 1

            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        

        if len(dqn_trainer.replay_buffer) > params.start_train_ts:
            dqn_trainer.train(ts)

        if ts % params.log_every == 0:
            out_str = "Timestep {}, Episode {}".format(ts, episodes_completed)
            if len(all_rewards) > 0:
                out_str += ", Last Reward: {}, Last 20 Avg Reward: {}, Last 20 Max: {}".format(all_rewards[-1], sum(all_rewards[-20:])/len(all_rewards[-20:]), max(all_rewards[-20:]))

            print(out_str)
    
    test_ensembles(candidate_ensembles, env, avf_trainers)
    #agent.save_weights(params.save_dir, params.max_ts)
    #F_trainer.save_weights(params.save_dir)


def test_ensembles(candidate_ensembles, env, avf_trainers):
    for i in range(len(candidate_ensembles)):
        agent_ids = [a.agent_id for a in candidate_ensembles[i]]
        """
        avfs = [avf_trainers[a.agent_id] for a in candidate_ensembles[i]]


        for j in range(len(avfs)):
            if not avfs[j].trained:
                avfs[j].train(verbose=False)
        """
        avfs = []
        #Temporary - add a bunch of experience to the avf buffer with a frozen policy and train
        for j in range(len(candidate_ensembles[i])):
            avf = train_avf(deepcopy(env), candidate_ensembles[i][j], 20)
            avfs.append(avf)
        ensemble = Ensemble(deepcopy(env), candidate_ensembles[i], avfs)

        ensemble_rollout = Rollout(ensemble, deepcopy(env))
        tot_rew = 0
        for i in range(20):
            rew = ensemble_rollout.validation_episode()
            tot_rew += rew
        tot_rew = tot_rew / 20
        
        print("Ensemble: {}. Reward: {}".format(agent_ids, tot_rew))
        


def pad_failures(failures, H):
    fail_values = np.zeros(len(failures))
    for i in range(len(failures)):
        failed = False
        for j in range(i+1, min(i+H+1, len(failures))): #Loop through next H transitions
            if failures[j] == 1:
                failed = True
                break
        if failed:
            fail_values[i] = 1
        else:
            fail_values[i] = failures[i]
    return fail_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=1400000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default='weights')
    parser.add_argument("--load_dir", type=str, default='weights')
    parser.add_argument("--save_transition", type=bool, default=True)
    parser.add_argument("--transition_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--transition_dir", type=str, default='transitions')
    parser.add_argument("--multiheaded", type=bool, default=False)
    parser.add_argument("--dropout", type=bool, default=False)
    parser.add_argument("--uncertainty", type=bool, default=False)
    parser.add_argument("--n_quantiles", type=int, default=50)
    parser.add_argument("--risk_estimation", type=bool, default=False)
    parser.add_argument("--tune_F_iters", type=int, default=10000)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    

    env = make_gym_env(args.env)
