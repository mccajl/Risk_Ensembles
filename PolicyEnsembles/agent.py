import random
import numpy as np
import torch
import os

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor


class Agent:
    def __init__(self, env, q_network, target_q_network, agent_id=None):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n
        self.agent_id = agent_id
        self.state_mean = None
        self.state_var = None

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

    def calculate_state_distribution(self, avf_trainer, rollout):
        
        # Use the replay buffer from the AVF: contains state information for the specific agent
        states = avf_trainer.replay_buffer.get_states()
        mean_state = np.mean(states, axis=0)
        var_state = np.var(states, axis=0)
        self.set_state_distribution(mean_state, var_state)
        """
        tot_rew = 0
        for k in range(50):
            tot_rew += rollout.validation_episode()
        state_vals, var_vals = rollout.get_state_distribution()
        """
    def set_state_distribution(self, mean, var):
        self.state_mean = mean
        self.state_var = var
    
    def get_state_distribution(self):
        return self.state_mean, self.state_var
    
    def save_weights(self, save_dir, step):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(self.q_network.state_dict(), save_dir+'/q_net_{}.pt'.format(step))
        torch.save(self.target_q_network.state_dict(), save_dir+'/target_net_{}.pt'.format(step))
    
    def load_weights(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(torch.load(path))
        #print("Agent Weights Loaded")

class DQN_Trainer:
    def __init__(self, agent, optimizer, replay_buffer, args):
        self.agent = agent
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.args = args

    def push_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, ts):
        # Update the q-network & the target network
        loss = self.compute_td_loss()
        if ts % self.args.target_network_update_f == 0:
            self.hard_update()


    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.batch_size)
        state = torch.tensor(np.float32(state)).type(dtype)
        next_state = torch.tensor(np.float32(next_state)).type(dtype)
        action = torch.tensor(action).type(dtypelong)
        reward = torch.tensor(reward).type(dtype)
        done = torch.tensor(done).type(dtype)

        # Normal DDQN update
        q_values = self.agent.q_network(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # double q-learning
        online_next_q_values = self.agent.q_network(next_state)
        _, max_indicies = torch.max(online_next_q_values, dim=1)
        target_q_values = self.agent.target_q_network(next_state)
        next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

        expected_q_value = reward + self.args.gamma * next_q_value.squeeze() * (1 - done)
        loss = (q_value - expected_q_value.data).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def soft_update(self, tau):
        for t_param, param in zip(self.agent.target_q_network.parameters(), self.agent.q_network.parameters()):
            if t_param is param:
                continue
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)
    
    def hard_update(self):
        for t_param, param in zip(self.agent.target_q_network.parameters(), self.agent.q_network.parameters()):
            if t_param is param:
                continue
            new_param = param.data
            t_param.data.copy_(new_param)

    
