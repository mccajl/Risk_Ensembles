#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python main.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000

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
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

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


def run_gym(params):
    if params.CnnDQN:
        env = make_atari(params.env)
        env = wrap_pytorch(wrap_deepmind(env))
        q_network = CnnDQN(env.observation_space.shape, env.action_space.n)
        target_q_network = deepcopy(q_network)
    else:
        env = make_gym_env(params.env)
        if params.dropout:
            q_network = DropoutDQN(env.observation_space.shape, env.action_space.n)
        else: 
            q_network = DQN(env.observation_space.shape, env.action_space.n)
        target_q_network = deepcopy(q_network)

    if USE_CUDA:
        q_network = q_network.cuda()
        target_q_network = target_q_network.cuda()

    for name, param in q_network.named_parameters():
            print(name, param.size())
            
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
    for ts in range(1, params.max_ts + 1):
        if params.save_transition and save_transitions:
            if transitions_left == 0:
                if len(episode_data['states']) > 10:
                    transitions.append(episode_data)
                save_transitions = False
                agent.save_weights(params.save_dir, ts-params.transition_steps)
                outfile = params.transition_dir + '/{}-{}-{}'.format(params.env, ts-params.transition_steps, int(sum(all_rewards[-20:])/len(all_rewards[-20:])))
                np.save(outfile, transitions)
                transitions = []
                print('Epsilon: {}'.format(epsilon))


        if params.save_transition and ts % params.save_every == 0:
            save_transitions = True
            transitions = []
            episode_data = {'states': [], 'actions': [], 'dones': [], 'rewards': [], 'q_values': []}
            transitions_left = params.transition_steps

        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )
        action, value = agent.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(int(action.cpu()))
        replay_buffer.push(state, action, reward, next_state, done)
        if save_transitions:
            transitions_left = transitions_left - 1
            episode_data['states'].append(state)
            episode_data['actions'].append(int(action.cpu()))
            episode_data['dones'].append(done)
            episode_data['rewards'].append(reward)
            episode_data['q_values'].append(value)
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            if save_transitions:
                if len(episode_data['actions']) > 10:
                    transitions.append(episode_data)
                episode_data = {'states': [], 'actions': [], 'dones': [], 'rewards': [], 'q_values': []}

        if len(replay_buffer) > params.start_train_ts:
            # Update the q-network & the target network
            loss = compute_td_loss(
                agent, params.batch_size, replay_buffer, optimizer, params.gamma
            )
            losses.append(loss.data)

            if ts % params.target_network_update_f == 0:
                hard_update(agent.q_network, agent.target_q_network)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Last Reward: {}, Last 20 Avg Reward: {}, Last 20 Max: {}".format(all_rewards[-1], sum(all_rewards[-20:])/len(all_rewards[-20:]), max(all_rewards[-20:]))
            #if len(losses) > 0:
                #out_str += ", TD Loss: {}".format(sum(losses[-params.log_every:])/params.log_every)
            if params.env == 'Traffic':
                wait_times = [s['total_wait_time'] for s in env.metrics]
                avg_wait_times = sum(wait_times) / len(wait_times)
                out_str += ', Avg Wait: {}'.format(avg_wait_times)
            print(out_str)
    
    agent.save_weights(params.save_dir, params.max_ts)


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
    run_gym(parser.parse_args())
