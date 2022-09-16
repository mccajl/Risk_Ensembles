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


agent_pairwise_linkage = {}



def get_agent_info(env, agent):
    avf = train_avf(env, agent, 20)
    roll = Rollout(agent, deepcopy(env))
    tot_rew = 0
    for k in range(20):
        tot_rew += roll.validation_episode()
    avg_rew = tot_rew / 20
    state_vals, var_vals = roll.get_state_distribution()
    all_states = roll.states_encountered

    return avf, avg_rew, state_vals, var_vals, all_states

def run_ensemble(env, ens):
    ens_roll = Rollout(ensemble, deepcopy(env))
    ens_rew = 0
    for j in range(20):
        ens_rew += ens_roll.validation_episode()
    ens_rew = ens_rew / 20
    return ens_rew

def complete_link(X, Y):
    max_dist = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            dist = np.linalg.norm(X[i] - Y[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist

def separation(X, Y):
    total_separation = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            dist = np.linalg.norm(X[i] - Y[j])
            total_separation += dist
    return total_separation

def pairwise_linkage(clusters, agents):
    tot_linkage = 0
    for i in range(len(clusters)):
        agent_i_id = agents[i].agent_id
        for j in range(len(clusters)):
            agent_j_id = agents[j].agent_id
            pair = tuple(sorted([agent_i_id, agent_j_id])) #Key to pairwise linkage dictionary
            if i != j:
                if pair not in agent_pairwise_linkage:               
                    cluster_linkage = complete_link(clusters[i], clusters[j])
                    agent_pairwise_linkage[pair] = cluster_linkage
                else:
                    cluster_linkage = agent_pairwise_linkage[pair]
                tot_linkage += cluster_linkage
    return tot_linkage

def pairwise_separation(clusters, agents):
    tot_separation = 0
    num_pairs = 0
    for i in range(len(clusters)):
        agent_i_id = agents[i].agent_id
        for j in range(len(clusters)):
            agent_j_id = agents[j].agent_id
            pair = tuple(sorted([agent_i_id, agent_j_id])) #Key to pairwise separation dictionary
            if i != j:
                if pair not in agent_pairwise_linkage:               
                    cluster_separation = separation(clusters[i], clusters[j])
                    agent_pairwise_linkage[pair] = cluster_separation
                else:
                    cluster_separation = agent_pairwise_linkage[pair]
                
                tot_separation += cluster_separation
                num_pairs += 1
    
    return tot_separation / num_pairs

def get_ens_variance(agents, state_vals):
    ens_state_vals = [state_vals[a.agent_id] for a in agents]
    ens_feat_var = np.var(np.array(ens_state_vals), axis=0)
    avg_state_var = np.mean(ens_feat_var)
    return avg_state_var

def form_objective(agents, state_vals, var_vals, agent_states):
    """
    state_var = get_ens_variance(agents, state_vals)
    ens_var_vals = [var_vals[a.agent_id] for a in agents]
    avg_agent_var = np.mean(np.array(ens_var_vals), axis=1)
    var_term = np.mean(avg_agent_var)
    objective = state_var + (0.0 * var_term)
    """
    ensemble_states = np.array([agent_states[a.agent_id] for a in agents])
    #objective = pairwise_linkage(ensemble_states, agents)
    objective = pairwise_separation(ensemble_states, agents)
    return objective

def generate_combinations(cur_ensemble, candidate_agents, n=4):
    all_agents = list(cur_ensemble) + list(candidate_agents)
    possible_ensembles = combinations(all_agents, n)
    return possible_ensembles

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
    all_encountered_states = {}
    #Initialize first 4 agents for ensemble

    ens_agents = []
    for i in range(4):  
        id_num = agent_nums[i]
        agent_i = agents[agent_nums[i]]
        ens_agents.append(agent_i)
        assert id_num == agent_i.agent_id
        agent_avf, agent_rew, agent_state_vals, agent_var_vals, all_states = get_agent_info(env, agent_i)
        agent_avfs[id_num] = agent_avf
        individual_agent_rew[id_num] = agent_rew
        agent_avg_state_vals[id_num] = agent_state_vals
        agent_state_variance[id_num] = agent_var_vals
        all_encountered_states[id_num] = np.array(all_states)
    

    ens_avfs = [agent_avfs[a.agent_id] for a in ens_agents]
    ensemble = Ensemble(deepcopy(env), ens_agents, ens_avfs)
    avg_ensemble_rew = run_ensemble(env, ensemble)
    ensemble_objective = form_objective(ens_agents, agent_avg_state_vals, agent_state_variance, all_encountered_states)

    best_ensemble_rew = avg_ensemble_rew

    total_overhead = 20 #Everytime an ensemble is tested, add 20 episodes to overhead

    candidates = []
    K = 10
    print("Ensemble Reward: {}".format(avg_ensemble_rew))
    print("Enemble Objective: {}".format(ensemble_objective))
    for i in range(4, len(agent_nums)):
        id_num = agent_nums[i]
        if len(candidates) < K:
            candidates.append(agents[id_num])
        else:
            possible_ensembles = generate_combinations(ens_agents, candidates)
            possible_ensembles = list(possible_ensembles)
            candidates = []
            replaced = False
            print("Checking {} Combos...".format(len(possible_ensembles)))
            for j, ens in enumerate(possible_ensembles):
                ens = list(ens)
                for k in range(len(ens)): #Fill agent information
                    test_agent = ens[k]
                    """
                    Right now, we train an AVF for every agent just because that's how we get rollouts.
                    In practice, we would not train AVF at this stage, only once a best ensemble has been chosen.
                    """
                    if test_agent.agent_id not in individual_agent_rew:
                        #Should only occur K times in each j loop
                        agent_avf, agent_rew, agent_state_vals, agent_var_vals, all_states = get_agent_info(env, test_agent)
                        agent_avfs[test_agent.agent_id] = agent_avf
                        individual_agent_rew[test_agent.agent_id] = agent_rew
                        agent_avg_state_vals[test_agent.agent_id] = agent_state_vals
                        agent_state_variance[test_agent.agent_id] = agent_var_vals
                        all_encountered_states[test_agent.agent_id] = np.array(all_states)
                test_objective = form_objective(ens, agent_avg_state_vals, agent_state_variance, all_encountered_states)
                if test_objective > ensemble_objective:
                    replaced = True
                    ensemble_objective = test_objective
                    ens_agents = deepcopy(ens)
            
            if replaced:
                print("Ensemble changed...")
                ens_avfs = [agent_avfs[a.agent_id] for a in ens_agents]
                ensemble = Ensemble(deepcopy(env), ens_agents, ens_avfs)
                avg_ensemble_rew = run_ensemble(env, ensemble)
                total_overhead += 20

            print("Ensemble Reward: {}".format(avg_ensemble_rew))
            print("Enemble Objective: {}".format(ensemble_objective))
            print("Current Overhead: {}".format(total_overhead))

            




    

    


