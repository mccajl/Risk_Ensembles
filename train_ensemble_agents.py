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
import time
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

    avf_trainers = {} #We use a separate risk estimation trainer for every checkpointed agent. Dictionary to hash by timesteps trained

    avf_epochs = params.avf_epochs

    episodes_completed = 0

    num_policies = params.num_policies
    save_every = int(params.max_ts / num_policies)
    avf_buffer_size = params.avf_buffer_size
    placeholder_buffer = Approx_Buffer(size=avf_buffer_size)
    current_agent_id = save_every
    ensemble_size = params.ensemble_n
    test_ensemble_num = params.ensembles_to_test
    
    average_rewards = []
    max_rewards = []

    assert avf_buffer_size / 2 < save_every, 'Functionality not implemented for replay buffers longer than checkpoint distance'

    candidate_agents = []


    for ts in range(1, params.max_ts + 1):
        
        if ts == current_agent_id:
            agent.save_weights(params.save_dir, ts)
            checkpoint_agent = Agent(env, deepcopy(agent.q_network), deepcopy(agent.target_q_network), agent_id=ts)
            candidate_agents.append(checkpoint_agent)
            


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

            """
            Pad sequences with failure (1 or 0) based on if agent fails within H timesteps
            """
            fail_padding = pad_failures(episode_failures, params.H)
            
            placeholder_buffer.push(np.array(states), fail_padding)


            if ts > (current_agent_id + (avf_buffer_size / 2)): # Finalize checkpointed agent avf buffer
                agent_buffer = deepcopy(placeholder_buffer)
                agent_avf_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]])), agent_buffer, training_iter=32)
                for i in range(avf_epochs):
                    agent_avf_trainer.train(verbose=False)
                avf_trainers[current_agent_id] = agent_avf_trainer
                current_agent_id = current_agent_id + save_every # Update current agent id to next agent checkpoint      

            states = []
            episode_failures = []

            episodes_completed += 1

            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

            average_rewards.append(np.mean(all_rewards))
            max_rewards.append(np.max(all_rewards))
        

        if len(dqn_trainer.replay_buffer) > params.start_train_ts:
            dqn_trainer.train(ts)

        if ts % params.log_every == 0:
            out_str = "Timestep {}, Episode {}".format(ts, episodes_completed)
            if len(all_rewards) > 0:
                out_str += ", Last Reward: {}, Last 20 Avg Reward: {}, Last 20 Max: {}".format(all_rewards[-1], sum(all_rewards[-20:])/len(all_rewards[-20:]), max(all_rewards[-20:]))

            print(out_str)
    

    if current_agent_id not in avf_trainers: #Catch final checkpointed agent
        agent_buffer = deepcopy(placeholder_buffer)
        agent_avf_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]])), agent_buffer, training_iter=32)
        for i in range(avf_epochs):
            agent_avf_trainer.train(verbose=False)
        avf_trainers[current_agent_id] = agent_avf_trainer
    
    best_ensembles = sort_ensembles(candidate_agents, ensemble_size, avf_trainers, test_ensemble_num)
    test_ensembles(deepcopy(env), best_ensembles, avf_trainers)


def sort_ensembles(agents, ensemble_n, avf_trainers, num_to_test):
    start = time.perf_counter()
    possible_ensembles = list(combinations(agents, ensemble_n))
    avg_state_values = {}
    ens_variance_by_index = {}
    for i, ens in enumerate(possible_ensembles):
        agent_avfs = [avf_trainers[a.agent_id] for a in ens]
        ens_agent_state_values = []
        #Calculate average state value for each agent in the ensemble
        for j, agent in enumerate(ens):
            if agent.agent_id in avg_state_values:
                agent_vals = avg_state_values[agent.agent_id]
            else:
                agent_vals = agent_avfs[j].state_averages()
                avg_state_values[agent.agent_id] = agent_vals
            ens_agent_state_values.append(agent_vals)
        
        ens_state_variance = np.var(np.array(ens_agent_state_values), axis=0) # Get variance among features for agents in ensemble
        mean_ens_variance = np.mean(ens_state_variance) # Average variance among features. Use this as a measure for ensemble diversity
        ens_variance_by_index[i] = mean_ens_variance
    
    #Sort ensembles by highest variance. Use the top num_to_test ensembles for testing
    sorted_diverse_ensembles = dict(sorted(ens_variance_by_index.items(), key=lambda item: item[1])) #Indices of most diverse ensembles, sorted
    test_ensemble_indices = list(sorted_diverse_ensembles.keys())[-num_to_test:]
    
    ensembles_to_test = np.array(possible_ensembles)[test_ensemble_indices]
    stop = time.perf_counter()
    minutes = (stop-start)/60
    print("Ensembles sorted in {:0.2f} minutes".format(minutes))
    return ensembles_to_test

    

def test_ensembles(env, ensembles, avf_trainers):
    start = time.perf_counter()
    rew_by_index = {}
    for i, ens in enumerate(ensembles):
        agent_ids = [a.agent_id for a in ens]
        agent_avfs = [avf_trainers[a.agent_id] for a in ens]
        ensemble = Ensemble(env, ens, agent_avfs)
        ens_rollout = Rollout(ensemble, deepcopy(env))
        tot_rew = 0
        for j in range(10): # Test each ensemble for 10 episodes to get performance. Results in 10 * num_to_test overhead episodes
            rew = ens_rollout.validation_episode()
            tot_rew += rew
        avg_rew = tot_rew / 10
        rew_by_index[i] = avg_rew

        print("Ensemble #{}: Agents {}, Reward {}".format(len(ensembles)-i, agent_ids, avg_rew))
    
    best_ensembles = dict(sorted(rew_by_index.items(), key=lambda item: item[1]))
    best_ensemble_index = list(best_ensembles.keys())[-1]
    best_ensemble = ensembles[best_ensemble_index]
    best_agents = [a.agent_id for a in best_ensemble]
    print("Best Ensemble:")
    print("Agents: {}".format(best_agents))
    print("Reward: {}".format(rew_by_index[best_ensemble_index]))
    stop = time.perf_counter()
    minutes = (stop-start)/60
    print("Best ensemble found in {:0.2f}".format(minutes))


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
