# A test file meant to test every possible ensemble from a file of weights
import os
from PolicyEnsembles.agent import Agent
from PolicyEnsembles.ensemble import Ensemble
from risk_estimation.risk_estimation import Rollout, Trainer, Approximator, Approx_Buffer
from DQN.helpers import make_atari, make_gym_env
from DQN.models import DQN
from copy import deepcopy
import argparse
from itertools import combinations
from best_frozen_ensembles.train_avfs import train_avf
import operator as op
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)

    folder = './weights/FullCart'

    agents = {}
    individual_agent_rew = {}
    agent_avg_state_vals = {}
    agent_nums = []
    for filename in os.listdir(folder):
        if filename.startswith('q'):
            f = os.path.join(folder, filename)
            string_vals = filename.split('_')
            agent_num = int(string_vals[2].split('.')[0])
            agent_nums.append(agent_num)
    
    agent_nums = sorted(agent_nums)
    for i in range(len(agent_nums)):
        agent_i = Agent(env, deepcopy(q_network), deepcopy(target_q_network), agent_id=agent_nums[i])
        agent_i.load_weights(folder+'/q_net_{}.pt'.format(agent_nums[i]))
        agents[agent_nums[i]] = agent_i


    #chosen_agents = agent_nums[20:len(agent_nums)-10:25]
    chosen_agents = agent_nums[:8]
    print(chosen_agents)
    possible_ensembles = combinations(chosen_agents, 4)
    num_combos = ncr(len(chosen_agents), 4)
    print("Testing {} ensembles".format(num_combos))
    rew_by_index = {}
    state_var_by_index = {}
    for i, ens in enumerate(possible_ensembles):
        ens_agents = [agents[j] for j in ens]
        avfs = []
        for j in range(len(ens_agents)):
            avf_j = train_avf(env, ens_agents[j], 20)
            avfs.append(avf_j)
        
        ens_agent_state_vals = []
        for j in range(len(ens_agents)):
            id_num = ens_agents[j].agent_id
            assert id_num == ens[j]
            if id_num not in individual_agent_rew:
                roll = Rollout(ens_agents[j], deepcopy(env))
                tot_rew = 0
                for k in range(10):
                    tot_rew += roll.validation_episode()
                individual_agent_rew[id_num] = tot_rew / 10
                agent_avg_state_vals[id_num] = roll.get_avg_state_value()
            ens_agent_state_vals.append(agent_avg_state_vals[id_num])
        
        #Calculate variance of on policy state values among agents in the ensemble

        ens_state_variance = np.var(np.array(ens_agent_state_vals), axis=0)
        state_var_by_index[i] = np.mean(ens_state_variance)

        #Then rollout ensemble reward
        single_rewards = [individual_agent_rew[j] for j in ens]
        print("Ensemble with agents: {} and rewards: {}".format(list(ens), single_rewards))
        print("State variance: {}".format(np.mean(ens_state_variance)))
        ensemble = Ensemble(env, ens_agents, avfs)
        ens_roll = Rollout(ensemble, deepcopy(env))
        ens_rew = 0
        for j in range(10):
            ens_rew += ens_roll.validation_episode()
        rew_by_index[i] = ens_rew / 10
        print("Reward: {}".format(rew_by_index[i]))
    
    possible_ensembles = list(possible_ensembles)

    sorted_ens_indices = dict(sorted(rew_by_index.items(), key=lambda item: item[1]))
    """
    best_twenty_indices = list(sorted_ens_indices.keys())[-20:]
    for i in range(len(best_twenty_indices)):
        print("#{} Ensemble: {}".format(20-i, possible_ensembles[best_twenty_indices[i]]))
        single_rewards = [individual_agent_rew[j] for j in possible_ensembles[best_twenty_indices[i]]]
        print("Individual Agent Rewards: {}".format(single_rewards))
        print("Ensemble Reward: {}".format(rew_by_index[best_twenty_indices[i]]))
        print("Ensemble variance: {}".format(state_var_by_index[best_twenty_indices[i]]))
    """
    
    """
    ens_ind_by_rew = list(sorted_ens_indices.keys())
    plot_rew = [rew_by_index[ind] for ind in ens_ind_by_rew]
    plot_var = [state_var_by_index[ind] for ind in ens_ind_by_rew]
    plot_rew = np.array(plot_rew)
    plot_var = np.array(plot_var)
    plot_rew = (plot_rew - np.min(plot_rew)) / (np.max(plot_rew) - np.min(plot_rew))
    plot_var = (plot_var - np.min(plot_var)) / (np.max(plot_var) - np.min(plot_var))
    x_vals = np.arange(len(plot_rew))
    plt.plot(x_vals, plot_rew, label='Ensemble reward')
    plt.plot(x_vals, plot_var, label='State variance')
    plt.title("Ensemble reward and variance, normalized")
    plt.legend()
    plt.savefig('./Ensemble_rew_vs_var_FULL.png')
    plt.show()
    plt.clf()
    """



    

    


