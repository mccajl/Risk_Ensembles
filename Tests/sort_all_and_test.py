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
    chosen_agents = agent_nums[:50]
    possible_ensembles = combinations(chosen_agents, 4)
    num_combos = ncr(len(chosen_agents), 4)
    print("Testing {} ensembles".format(num_combos))
    rew_by_index = {}
    state_var_by_index = {}
    agent_avfs = {}
    possible_ensembles = list(possible_ensembles)
    for i, ens in enumerate(possible_ensembles):
        ens_agents = [agents[j] for j in ens]
        avfs = []
        for j in range(len(ens_agents)):
            if ens_agents[j].agent_id not in agent_avfs:
                avf_j = train_avf(env, ens_agents[j], 20)
                agent_avfs[ens_agents[j].agent_id] = avf_j
            else:
                avf_j = agent_avfs[ens_agents[j].agent_id]
            avfs.append(avf_j)
        
        ens_agent_state_vals = []
        for j in range(len(ens_agents)):
            id_num = ens_agents[j].agent_id
            assert id_num == ens[j]
            if id_num not in individual_agent_rew:
                roll = Rollout(ens_agents[j], deepcopy(env))
                tot_rew = 0
                for k in range(20):
                    tot_rew += roll.validation_episode()
                individual_agent_rew[id_num] = tot_rew / 20
                agent_avg_state_vals[id_num] = roll.get_avg_state_value()
                print("Agent {} Reward: {}".format(id_num, tot_rew/20))
            ens_agent_state_vals.append(agent_avg_state_vals[id_num])
        
        #Calculate variance of on policy state values among agents in the ensemble

        ens_state_variance = np.var(np.array(ens_agent_state_vals), axis=0)
        state_var_by_index[i] = np.mean(ens_state_variance)

    most_diverse_ensembles = dict(sorted(state_var_by_index.items(), key=lambda item: item[1]))

    top_ensembles = list(most_diverse_ensembles.keys())[-20:]
    all_rewards = []
    for k in range(len(top_ensembles)):
        ens = possible_ensembles[top_ensembles[k]]
        ens_agents = [agents[i] for i in ens]
        ens_avfs = [agent_avfs[a.agent_id] for a in ens_agents]
        ens_agent_state_vals = [agent_avg_state_vals[a.agent_id] for a in ens_agents]
        ens_state_variance = np.mean(np.var(np.array(ens_agent_state_vals), axis=0))
        ensemble = Ensemble(env, ens_agents, ens_avfs)
        ens_roll = Rollout(ensemble, deepcopy(env))
        ens_rew = 0
        for j in range(10):
            ens_rew += ens_roll.validation_episode()
        ens_rew = ens_rew / 10
        all_rewards.append(ens_rew)
        print("#{}: Ensemble: {}, Reward: {}, Variance: {}".format(20-k, ens, ens_rew, ens_state_variance))
        print(ensemble.get_counts())

    #n, bins, patches = plt.hist(all_rewards, bins=10, range=(0, 10000))
    #plt.title('Histogram of Reward Among 50 Most Diverse Ensembles')
    #plt.xlabel('Reward')
    #plt.ylabel('Frequency')
    #plt.savefig('./Cart_hist_reward.png')




    

    


