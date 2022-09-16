# Failure prediction network implementation based off of Rigorous Agent Evaluation: An Adversarial
# Approach to Uncover Catastrophic Failures, Uesato et al 2018.

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical
import random

class Approximator(nn.Module):
    #Approximator network takes in a state instance and predicts the probability of agent failure from this state
    #Treated as binary classification
    def __init__(self, input_shape):
        super(Approximator, self).__init__()
        self.input_shape = input_shape
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_shape[0], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class Approx_Buffer(object):
    #Buffer class for the failure search approximator
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)
        self.states = []
    
    def push(self, states, results):
        #At the end of each training episode, we push the states for the entire episode into the buffer,
        #along with a binary sequence indicating if the agent fails within H steps of the transition
        for i in range(len(states)):
            state = np.expand_dims(states[i], 0)
            self.states.append(state)
            result = np.reshape(results[i], [1, -1])
            self.buffer.append((state, result))

    def sample(self, batch_size):
        states, results = zip(
            *random.sample(self.buffer, batch_size)
        )
        states = np.concatenate(states)
        results = np.concatenate(results)
        states = torch.from_numpy(states).float()
        results = torch.from_numpy(results).float()
        return states, results

    def get_states(self):
        return np.array(self.states)
    
    def get_state_averages(self):
        states = np.array(self.states)
        return np.mean(states, axis=0)
    
    def __len__(self):
        return len(self.buffer)



class Trainer(object):
    #Trainer class for the AVF
    def __init__(self, network, replay_buffer, lr=0.0001, batch_size=32, training_iter=32):
        self.network = network
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.batch_size = batch_size
        self.training_iter = training_iter #How many iterations to train on each time we train the approximator
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()
        #self.loss_fn = nn.MSELoss()
        self.trained = False
    

    def add_experience(self, states, result):
        self.replay_buffer.push(states, result)
    
    def train_step(self, states, results):
        #Training step for a single batch: Take in batch size states and result of trajectory (failures or success)
        #All the inputs should be pytorch tensors

        if results.shape == (self.batch_size):
            results = torch.unsqueeze(results, 1)
        
        predictions = self.network(states)
        loss = self.loss_fn(predictions, results)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        predictions = predictions.reshape(-1).detach().numpy()
        #print("output mean ", np.mean(predictions))
        class_assignments = predictions.round()
        
        class_ground_truths = results.reshape(-1).detach().numpy()
        #print("label mean ", np.mean(class_ground_truths))
        accuracy = (class_assignments == class_ground_truths).mean()

        return loss, accuracy
    
    def train(self, verbose=True):
        total_loss = []
        total_accuracy = []
        for i in range(self.training_iter):
            states, results = self.replay_buffer.sample(self.batch_size)
            loss, accuracy = self.train_step(states, results)
            total_loss.append(loss)
            total_accuracy.append(accuracy)

        if verbose:
            print("Approximator Loss: {} Accuracy: {}".format(sum(total_loss)/len(total_loss),
                                                              sum(total_accuracy)/len(total_accuracy)))
        self.trained = True
        
    def predict(self, x): #Single network prediction on an input. Not rounded
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.network(x).reshape(-1).detach().numpy().item()
    
    def eval_state(self, state):
        state = np.reshape(state, [1, -1])
        risk = self.predict(state)
        return risk
        
    def evaluation(self, states, results, print_values=False): #Give N episodes worth of data to test the model on
        state_size = len(states[0])
        states = np.reshape(states, [-1, state_size])
        inputs = torch.from_numpy(states).float()
        predictions = self.network(inputs)
        predictions = predictions.reshape(-1).detach().numpy()
        if print_values:
            print("Raw prediction values: {}".format(predictions))
        results = np.reshape(results, [-1, 1])
        class_assignments = predictions.round()
        accuracy = (class_assignments == results).mean()
        return accuracy, np.mean(predictions)

    def state_averages(self):
        return self.replay_buffer.get_state_averages()
        
    def save_weights(self, save_dir):
        torch.save(self.network.state_dict(), save_dir+'/risk_estimator_fixed_underfit.pt')
    
    def load_weights(self, save_dir):
        self.network.load_state_dict(torch.load(save_dir+'/risk_estimator_fixed_underfit.pt'))





class Rollout(object):
    """
    Class for rolling out a single episode to failure, from a start point
    params:
    agent - has an 'act' method that takes state and epsilon as input, outputs action
    env - has a 'reset_state' method that takes in a valid state, and resets the episode, placing
    the agent at a given state. Also has a 'step' method which returns an info variable, where
    info['failure'] is 1 if the agent failed in that timestep, and 0 else.
    """
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.states_encountered = []

    def evaluate(self, start_state=None):
        #Evaluates the agent from the start_state. Returns 1 if the agent fails, 0 else.
        state = self.env.reset_state(start_state)
        done = False
        while not done:
            action, _ = self.agent.act(state, epsilon=0)
            if torch.is_tensor(action):
                action = action.detach().numpy().item()
            next_state, r, done, info = self.env.step(action)
            if info['failure'] == 1:
                return 1
            
            state = next_state
        
        return 0

    def validation_episode(self):
        state = self.env.reset()
        done = False
        total_r = 0
        while not done:
            self.states_encountered.append(state)
            action, _ = self.agent.act(state, epsilon=0)
            if torch.is_tensor(action):
                action = action.detach().numpy().item()
            next_state, r, done, info = self.env.step(action)
            total_r += r
            
            state = next_state
        
        return total_r
    
    def episode_with_avf_info(self):
        states = []
        fails = []
        state = self.env.reset()
        done = False
        total_r = 0
        while not done:
            states.append(state)
            action, _ = self.agent.act(state, epsilon=0)
            if torch.is_tensor(action):
                action = action.detach().numpy().item()
            next_state, r, done, info = self.env.step(action)
            fails.append(info['failure'])
            total_r += r
            
            state = next_state
        
        return total_r, states, fails
    
    def get_avg_state_value(self):
        states = np.array(self.states_encountered)
        return np.mean(states, axis=0)
    
    def get_state_variance(self):
        states = np.array(self.states_encountered)
        return np.var(states, axis=0)
    
    def get_state_distribution(self):
        mean = self.get_avg_state_value()
        var = self.get_state_variance()
        self.agent.set_state_distribution(mean, var) #Set distribution in the agent class
        return mean, var





