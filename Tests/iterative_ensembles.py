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

def get_agent_info(env, agent):
    avf = train_avf(env, agent, 20)
    roll = Rollout(agent, deepcopy(env))
    tot_rew = 0
    for k in range(50):
        tot_rew += roll.validation_episode()
    avg_rew = tot_rew / 50
    state_vals, var_vals = roll.get_state_distribution()

    return avf, avg_rew, state_vals, var_vals

def run_ensemble(env, ens):
    ens_roll = Rollout(ensemble, deepcopy(env))
    ens_rew = 0
    for j in range(20):
        ens_rew += ens_roll.validation_episode()
    ens_rew = ens_rew / 20
    return ens_rew

def get_ens_variance(agents, state_vals):
    ens_state_vals = [state_vals[a.agent_id] for a in agents]
    ens_feat_var = np.var(np.array(ens_state_vals), axis=0)
    avg_state_var = np.mean(ens_feat_var)
    return avg_state_var

def form_objective(agents, state_vals, var_vals):
    state_var = get_ens_variance(agents, state_vals)
    ens_var_vals = [var_vals[a.agent_id] for a in agents]
    avg_agent_var = np.mean(np.array(ens_var_vals), axis=1)
    var_term = np.mean(avg_agent_var)
    objective = state_var + (0.5 * var_term)
    return objective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)

    folder = './weights/CartEnsembleVal'

    agents = {}
    agent_nums = []
    for filename in os.listdir(folder):
        if filename.startswith('q'):
            f = os.path.join(folder, filename)
            string_vals = filename.split('_')
            agent_num = int(string_vals[2].split('.')[0])
            agent_nums.append(agent_num)
    
    for i in range(len(agent_nums)):
        agent_i = Agent(env, deepcopy(q_network), deepcopy(target_q_network), agent_id=agent_nums[i])
        agent_i.load_weights(folder+'/q_net_{}.pt'.format(agent_nums[i]))
        agents[agent_nums[i]] = agent_i


    print("Testing {} Agents".format(len(agents)))


    #Dictionaries for agent information:
    agent_avfs = {} #Note that an AVF is only trained once per agent - this could have large effects on agent performance
    individual_agent_rew = {}
    agent_avg_state_vals = {}
    agent_state_variance = {}

    #Initialize first 4 agents for ensemble

    ens_agents = []
    for i in range(4):  
        id_num = agent_nums[i]
        agent_i = agents[agent_nums[i]]
        ens_agents.append(agent_i)
        assert id_num == agent_i.agent_id
        agent_avf, agent_rew, agent_state_vals, agent_var_vals = get_agent_info(env, agent_i)
        agent_avfs[id_num] = agent_avf
        individual_agent_rew[id_num] = agent_rew
        agent_avg_state_vals[id_num] = agent_state_vals
        agent_state_variance[id_num] = agent_var_vals
    

    ens_avfs = [agent_avfs[a.agent_id] for a in ens_agents]
    ensemble = Ensemble(deepcopy(env), ens_agents, ens_avfs, use_agent_info=False)
    avg_ensemble_rew = run_ensemble(env, ensemble)
    ensemble_objective = form_objective(ens_agents, agent_avg_state_vals, agent_state_variance)

    best_ensemble_rew = avg_ensemble_rew

    total_overhead = 20 #Everytime an ensemble is tested, add 20 episodes to overhead

    for i in range(4, len(agent_nums)):
        print("Considering agent {} ({})".format(agent_nums[i], i))
        print("Current Ensemble Objective: {}".format(ensemble_objective))
        print("Current Ensemble Reward: {}".format(avg_ensemble_rew))
        id_num = agent_nums[i]
        current_agent = agents[id_num]
        assert id_num == current_agent.agent_id

        #Update agent information - Should only happen once per agent
        agent_avf, agent_rew, agent_state_vals, agent_var_vals = get_agent_info(env, current_agent)
        agent_avfs[id_num] = agent_avf
        individual_agent_rew[id_num] = agent_rew
        agent_avg_state_vals[id_num] = agent_state_vals
        agent_state_variance[id_num] = agent_var_vals

        #Create 4 separate test ensembles - remove each agent in favor of current
        replace = -1
        highest_obj = ensemble_objective
        for j in range(4):
            test_ens_agents = deepcopy(ens_agents)
            test_ens_agents[j] = current_agent
            test_ens_avfs = [agent_avfs[a.agent_id] for a in test_ens_agents]
            test_ensemble = Ensemble(deepcopy(env), test_ens_agents, test_ens_avfs, use_agent_info=False)
            test_ensemble_obj = form_objective(test_ens_agents, agent_avg_state_vals, agent_state_variance)

            if test_ensemble_obj > highest_obj:
                replace = j
                highest_obj = test_ensemble_obj
        
        #Test new ensemble's reward
        if replace != -1:
            total_overhead += 20
            print("Current Overhead: {}".format(total_overhead))
            test_ens_agents = deepcopy(ens_agents)
            test_ens_agents[replace] = current_agent
            test_ens_avfs = [agent_avfs[a.agent_id] for a in test_ens_agents]
            test_ensemble = Ensemble(deepcopy(env), test_ens_agents, test_ens_avfs, use_agent_info=False)
            test_avg_ensemble_rew = run_ensemble(env, test_ensemble)

            if test_avg_ensemble_rew <= avg_ensemble_rew:
                replace = -1 #Don't replace if reward isn't improved
            else:
                avg_ensemble_rew = test_avg_ensemble_rew

        
        #Create new ensemble with new agent
        if replace != -1:
            ens_agents[replace] = current_agent
            ens_avfs = [agent_avfs[a.agent_id] for a in ens_agents]
            ensemble = Ensemble(deepcopy(env), ens_agents, ens_avfs, use_agent_info=False)
            ensemble_objective = form_objective(ens_agents, agent_avg_state_vals, agent_state_variance)

    print("Final Ensemble Objective: {}".format(ensemble_objective))
    print("Final Ensemble Reward: {}".format(avg_ensemble_rew))
    print("Final Overhead Episodes for Ensemble Testing: {}".format(total_overhead))

            




    

    


