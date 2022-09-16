# A test file meant to test every possible ensemble from a file of weights
import os
from PolicyEnsembles.agent import Agent
from PolicyEnsembles.ensemble import Ensemble
from risk_estimation.risk_estimation import Rollout, Trainer, Approximator, Approx_Buffer
from DQN.helpers import make_atari, make_gym_env
from DQN.models import DQN
from copy import deepcopy
import argparse
from best_frozen_ensembles.train_avfs import train_avf
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)

    folder = './weights/CartEnsembleVal'

    agents = []
    individual_agent_rew = []

    agent_nums = [1813, 1931, 2227, 2543]
    
    for i in range(len(agent_nums)):
        agent_i = Agent(env, deepcopy(q_network), deepcopy(target_q_network), agent_id=agent_nums[i])
        agent_i.load_weights(folder+'/q_net_{}.pt'.format(agent_nums[i]))
        agents.append(agent_i)


    avfs = []
    for j in range(len(agents)):
        avf_j = train_avf(env, agents[j], 20)
        avfs.append(avf_j)
    
    state_values = []
    for j in range(len(agents)):
        roll = Rollout(agents[j], deepcopy(env))
        tot_rew = 0
        for k in range(30):
            tot_rew += roll.validation_episode()
        individual_agent_rew.append(tot_rew / 30)
        state_values.append(roll.get_avg_state_value())
    state_values = np.array(state_values)
    state_value_var = np.var(state_values, axis=0)
    #Then rollout ensemble reward
    print("Ensemble with agents: {} and rewards: {}".format(agent_nums, individual_agent_rew))
    print("Agent avg state values: {}".format(state_values))
    print("Agent state value variance: {}".format(state_value_var))
    print("Avg cart variance: {}".format(np.mean(state_value_var[:2])))
    print("Avg pole variance: {}".format(np.mean(state_value_var[2:])))
    print("State value var averaged: {}".format(np.mean(state_value_var)))
    ensemble = Ensemble(env, agents, avfs, use_agent_info=False)
    ens_roll = Rollout(ensemble, deepcopy(env))
    ens_rew = 0
    for j in range(20):
        ens_rew += ens_roll.validation_episode()
    ens_rew = ens_rew / 20
    print("Reward: {}".format(ens_rew))
    print("Agent Action Frequencies: {}".format(ensemble.get_counts()))
    print("Agent Average Risk: {}".format(ensemble.get_risks()))
    print("Average action variance among agents: {}".format(ensemble.get_act_variance()))


    

    


