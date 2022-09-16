#!/usr/bin/env python3

"""
Usage:

Example cartpole command (~8k ts to solve):
    python main.py \
        --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
        --replay_size 5000 --starts_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
        --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
"""


#TRAINING FILE FOR A SINGLE FAILURE PREDICTION NETWORK, PARAMETERIZED BASED ON TRAINING ITERATION
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

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor


class Agent:
    def __init__(self, env, q_network, target_q_network, agent_id=None):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n
        self.agent_id = agent_id

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
        q_value = self.q_network.forward(state)
        if random.random() > epsilon:
            q = q_value[0].detach().cpu().numpy()
            return q_value.max(1)[1].data[0], q
        else:
            action = random.randrange(self.env.action_space.n)
            q = q_value[0].detach().cpu().numpy()
            return torch.tensor(action), q

    def save_weights(self, save_dir, step):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(self.q_network.state_dict(), save_dir+'/q_net_{}.pt'.format(step))
        torch.save(self.target_q_network.state_dict(), save_dir+'/target_net_{}.pt'.format(step))
    
    def load_weights(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(torch.load(path))

def compute_td_loss(agent, batch_size, replay_buffer, optimizer, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).type(dtype)
    next_state = torch.tensor(np.float32(next_state)).type(dtype)
    action = torch.tensor(action).type(dtypelong)
    reward = torch.tensor(reward).type(dtype)
    done = torch.tensor(done).type(dtype)

    # Normal DDQN update
    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # double q-learning
    online_next_q_values = agent.q_network(next_state)
    _, max_indicies = torch.max(online_next_q_values, dim=1)
    target_q_values = agent.target_q_network(next_state)
    next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

    expected_q_value = reward + gamma * next_q_value.squeeze() * (1 - done)
    loss = (q_value - expected_q_value.data).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


def run_gym_estimation(params):
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


    F_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]+1])), Approx_Buffer()) #+1 for the training iteration (current training episode #)

            
    env.seed(params.seed)
    agent = Agent(env, q_network, target_q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    replay_buffer = ReplayBuffer(params.replay_size)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()

    if params.save_transition:
        if not os.path.exists(params.transition_dir):
            os.makedirs(params.transition_dir, exist_ok=True)
    save_transitions = False
    num_training_iterations = 0
    states = []
    agent_infos = []
    episode_failures = []

    ensemble_rewards = {}
    single_agent_rewards = {}

    saved_agents = []

    for ts in range(1, params.max_ts + 1):

        if ts % params.save_every == 0:
            agent.save_weights(params.save_dir, len(all_rewards)) #Save with episode iteration (for testing with AVF)
            saved_agents.append(len(all_rewards))
            single_agent_rewards[len(all_rewards)] = test_single_agent(params, len(all_rewards), deepcopy(env))


        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )
        action, value = agent.act(state, epsilon)
        

        next_state, reward, done, info = env.step(int(action.cpu()))

        states.append(state)
        agent_infos.append(len(all_rewards)) #Current training episode iteration
        assert 'failure' in info
        episode_failures.append(info['failure'])

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            """
            Pad sequences with failure (1 or 0) based on if agent fails within H timesteps
            """
            #fail_padding = pad_failures(episode_failures, params.H)
            fail_padding = pad_failures_MC(episode_failures, params.H)
            F_trainer.add_experience(np.array(states), np.array(agent_infos), fail_padding)
            states = []
            agent_infos = []
            episode_failures = []

            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        
        if (ts+1) % 10000 == 0:
            chosen_agents = np.random.choice(saved_agents, size=params.ensemble_n) #Random agent selection
            #chosen_agents = saved_agents[-(params.ensemble_n):] #Most recent n agents
            #sorted_rew = dict(sorted(single_agent_rewards.items(), key=lambda item: item[1]))
            #chosen_agents = list(sorted_rew.keys())[-(params.ensemble_n):] #Best n agents by reward
            ensemble_r = run_ensemble(params, chosen_agents, env)
            ensemble_rewards[len(all_rewards)] = ensemble_r

        if len(replay_buffer) > params.start_train_ts:
            # Update the q-network & the target network
            loss = compute_td_loss(
                agent, params.batch_size, replay_buffer, optimizer, params.gamma
            )
            losses.append(loss.data)
            num_training_iterations += 1
            if ts % params.target_network_update_f == 0:
                hard_update(agent.q_network, agent.target_q_network)
            

            if ts % params.tune_F_iters == 0 and len(F_trainer.replay_buffer) > F_trainer.batch_size:
                F_trainer.train(verbose=False)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Last Reward: {}, Last 20 Avg Reward: {}, Last 20 Max: {}".format(all_rewards[-1], sum(all_rewards[-20:])/len(all_rewards[-20:]), max(all_rewards[-20:]))

            print(out_str)
    
    #agent.save_weights(params.save_dir, params.max_ts)
    #F_trainer.save_weights(params.save_dir)

    plot_info(all_rewards, ensemble_rewards)

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

def pad_failures_MC(failures, H):
    fail_values = np.ones(len(failures))
    for i in range(len(failures)):
        success = False
        for j in range(i+1, min(i+H+1, len(failures))): #Loop through next H transitions
            if failures[j] == 0:
                success = True
                break
        if success:
            fail_values[i] = 0
        else:
            fail_values[i] = failures[i]
    return fail_values


def test_single_agent(args, chosen_agent, env):
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    agent = Agent(env, q_network, target_q_network, agent_id = chosen_agent)
    agent.load_weights(args.save_dir+'/q_net_{}.pt'.format(chosen_agent))
    rew = roll_single_agent(env, agent)
    return rew

def run_ensemble(args, chosen_agents, env):
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    agents = []
    for i in range(len(chosen_agents)):
        agent_i = Agent(env, deepcopy(q_network), deepcopy(target_q_network), agent_id=chosen_agents[i])
        agent_i.load_weights(args.save_dir+'/q_net_{}.pt'.format(chosen_agents[i]))
        agents.append(agent_i)
    avfs = []
    for i in range(len(agents)):
        avf = train_avf(deepcopy(env), agents[i], args.H)
        avfs.append(avf)
    
    print("Testing agents {}".format(chosen_agents))
    avg_rew = roll_ensemble(deepcopy(env), agents, avfs)
    return avg_rew

def plot_info(all_rewards, val_rewards):
    val_rew_arr = np.zeros(len(all_rewards))
    last_k = 0
    for k in val_rewards:
        for i in range(last_k, min(k+1, len(all_rewards))):
            val_rew_arr[i] = val_rewards[k]
        last_k = k+1
        if last_k >= len(all_rewards):
            break
    
    x = np.arange(len(all_rewards))
    plt.plot(x, all_rewards)
    plt.plot(x, val_rew_arr, 'bo')
    plt.ylabel('Reward')
    plt.xlabel('Training Episodes')
    plt.savefig('Cart_ensemble_val_random_N10.png')
    plt.show()

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
