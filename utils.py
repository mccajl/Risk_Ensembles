import numpy as np
import math
import argparse
from train import Agent
from train_multiheaded import MH_Agent
from DQN.helpers import make_gym_env
from DQN.models import DQN, MultiHeadDQN, DropoutDQN
from CAPS.data import Data
import torch

def env_uncertainty_counts(env, weights, transitions):
    nS = env.observation_space.shape
    nA = env.action_space.n
    agent1 = Agent(env, DQN(nS, nA), DQN(nS, nA))
    agent1.load_weights(weights[0])
    agent2 = Agent(env, DQN(nS, nA), DQN(nS, nA))
    agent2.load_weights(weights[1])
    agent3 = Agent(env, DQN(nS, nA), DQN(nS, nA))
    agent3.load_weights(weights[2])
    data1 = Data(transitions[0], nA)
    data2 = Data(transitions[1], nA)
    data3 = Data(transitions[2], nA)

    freq1 = gridworld_frequencies(data1)
    freq2 = gridworld_frequencies(data2)
    freq3 = gridworld_frequencies(data3)
    uncertain1 = normalized_gridworld_uncertainties(agent1)
    uncertain2 = normalized_gridworld_uncertainties(agent2)
    uncertain3 = normalized_gridworld_uncertainties(agent3)
    
    print("Frequencies 50000 steps: ")
    print(freq1)
    print("Certainties 50000 steps: ")
    print(1 - uncertain1)
    
    print("Frequencies 100000 steps: ")
    print(freq2)
    print("Certainties 100000 steps: ")
    print(1 - uncertain2)

    print("Frequencies 450000 steps: ")
    print(freq3)
    print("Certainties 450000 steps: ")
    print(1 - uncertain3)

def gridworld_frequencies(D):
    f = np.zeros(48)
    ns, nsa, nsas = D.state_frequencies(one_hot=True)
    N = 0
    for i in range(48):
        if tuple([i]) in ns:
            N += ns[tuple([i])]
            f[i] = ns[tuple([i])]
    f = f / N
    f = (f-min(f)) / (max(f) - min(f))
    return f

def normalized_gridworld_uncertainties(agent):
    uncertainties = np.zeros(48)
    for i in range(48):
        state = np.zeros(48)
        state[i] = 1
        _, q_vals = agent.act(state, 0)
        q_vals = np.array(q_vals)
        q_vals = (q_vals - min(q_vals)) / (max(q_vals) - min(q_vals))
        uncertainties[i] = evenness(q_vals)

    uncertainties = (uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))
    return uncertainties


def evenness(P):
    N = len(P)
    E = 0
    for pi in P:
        if pi != 0:
            E += pi * math.log(pi) / math.log(N)
    
    E = E * -1
    return E

def dropout_uncertainty(agent, state, K=20):
    #Calculate stdev of the value function for K forward passes of a dropout, stochastic network
    #May change to be mean taken along actions of the variance of K forward passes
    estimates = []
    for i in range(K):
        _, q = agent.act(state, 0)
        estimates.append(q)
    #estimates = (estimates - np.min(estimates)) / (np.max(estimates) - np.min(estimates))
    return np.std(estimates), np.mean(estimates)

def multihead_uncertainty(agent, state):
    #Calculate stdev of the value function of H heads of a multiheaded DQN. Uncertainty arises from lack of experience in a state
    estimates = []
    q_values = agent.q_network.forward(state)
    q_s_a = [[]] * agent.num_actions
    for i in range(agent.num_actions):
        for q_val in q_values:
            q_s_a[i].append(torch.squeeze(q_val.data).detach().cpu().numpy()[i])
    
    var_a = [math.pow(np.std(q),2) for q in q_s_a]
    uncertainty = sum(var_a) / len(var_a)
    return uncertainty

def env_uncertainty_multihead(agent):
    uncertainties = np.zeros(48)

    for i in range(48):
        state = np.zeros(48)
        state[i] = 1
        uncertainties[i] = multihead_uncertainty(agent, state)

    #uncertainties = (uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))
    return uncertainties

def env_uncertainty_dropout(agent):
    uncertainties = np.zeros(48)
    means = np.zeros(48)
    for i in range(48):
        state = np.zeros(48)
        state[i] = 1
        uncertainties[i], means[i] = dropout_uncertainty(agent, state)
        
    #uncertainties = (uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))
    return uncertainties, means

if __name__ == '__main__':
    #transition1 = np.load('transitions/cliffwalk/CliffWalking-50000--2010.npy', allow_pickle=True)
    #transition2 = np.load('transitions/cliffwalk/CliffWalking-100000--545.npy', allow_pickle=True)
    #transition3 = np.load('transitions/cliffwalk/CliffWalking-450000--13.npy', allow_pickle=True)
    #weight1 = 'weights/cliffwalk/q_net_50500.pt'
    #weight2 = 'weights/cliffwalk/q_net_100500.pt'
    #weight3 = 'weights/cliffwalk/q_net_450500.pt'

    env = make_gym_env('CliffWalking')
    nS = env.observation_space.shape
    nA = env.action_space.n
    
    #env_uncertainty_counts(env, [weight1, weight2, weight3], [transition1, transition2, transition3])
    

    #multihead_weight = 'weights/cliffwalk_multihead/'
    #mh_agent = MH_Agent(env, MultiHeadDQN(nS, nA), MultiHeadDQN(nS, nA))
    #mh_agent.load_weights(multihead_weight)
    #mh_uncertainties = env_uncertainty_multihead(mh_agent)
    #print(mh_uncertainties)

    print("50000 Steps")
    dropout_weight = 'weights/cliffwalk_dropout/q_net_50000.pt'
    drop_agent = Agent(env, DropoutDQN(nS, nA), DropoutDQN(nS, nA))
    drop_agent.load_weights(dropout_weight)
    drop_uncertainties, drop_means = env_uncertainty_dropout(drop_agent)
    print(drop_uncertainties[:12])
    print(drop_uncertainties[12:24])
    print(drop_uncertainties[24:36])
    print(drop_uncertainties[36:])
    print()
    print(drop_means[:12])
    print(drop_means[12:24])
    print(drop_means[24:36])
    print(drop_means[36:])
    
    print()
    print("400000 Steps")
    dropout_weight = 'weights/cliffwalk_dropout/q_net_400000.pt'
    drop_agent = Agent(env, DropoutDQN(nS, nA), DropoutDQN(nS, nA))
    drop_agent.load_weights(dropout_weight)
    drop_uncertainties, drop_means = env_uncertainty_dropout(drop_agent)
    print(drop_uncertainties[:12])
    print(drop_uncertainties[12:24])
    print(drop_uncertainties[24:36])
    print(drop_uncertainties[36:])
    print()
    print(drop_means[:12])
    print(drop_means[12:24])
    print(drop_means[24:36])
    print(drop_means[36:])





