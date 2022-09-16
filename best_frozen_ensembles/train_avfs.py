import torch
import numpy as np
import argparse
from PolicyEnsembles.agent import Agent
from PolicyEnsembles.ensemble import Ensemble
from risk_estimation.risk_estimation import Rollout, Approximator, Approx_Buffer, Trainer
from DQN.helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from DQN.models import DQN, CnnDQN, DropoutDQN
from copy import deepcopy


def fill_buffer(agent, env, AVF, n, H, mc=False):
    states = []
    agent_infos = []
    failures = []
    ep_reward = np.zeros(n)
    for i in range(n):
        done = False
        state = env.reset()
        episode_failures = []
        while not done:
            states.append(state)
            agent_infos.append(agent.agent_id)
            action, _ = agent.act(state, 0)
            if torch.is_tensor(action):
                action = action.detach().numpy().item()
            next_state, reward, done, info = env.step(action)
            ep_reward[i] += reward
            episode_failures.append(info['failure'])
            state = next_state
        
        if mc:
            failures = failures + list(pad_failures_MC(episode_failures, H))
        else:
            failures = failures + list(pad_failures(episode_failures, H))
            
    return np.array(states), np.array(agent_infos), np.array(failures)
    
    
    

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


def train_avf(env, agent, H):
    avf = Trainer(Approximator(tuple([env.observation_space.shape[0]])), Approx_Buffer(), training_iter=32)

    
    states, agent_infos, failures = fill_buffer(agent, env, avf, 200, H=H) #Fill to around full replay buffer (100k)
    avf.add_experience(states, failures)

    num_epochs = 30
    for j in range(num_epochs):
        avf.train(verbose=False)
    
    return avf



