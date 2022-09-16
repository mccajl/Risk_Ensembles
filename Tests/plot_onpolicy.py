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
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)

    folder = './weights/CartEnsembleFull'

    agents = []
    individual_agent_rew = []

    agent_nums = [2100, 4400, 4775, 5500]
    
    for i in range(len(agent_nums)):
        agent_i = Agent(env, deepcopy(q_network), deepcopy(target_q_network), agent_id=agent_nums[i])
        agent_i.load_weights(folder+'/q_net_{}.pt'.format(agent_nums[i]))
        agents.append(agent_i)


    avfs = []
    for j in range(len(agents)):
        avf_j = train_avf(env, agents[j], 20)
        avfs.append(avf_j)
    
    state_values = []
    agent_states = []
    for j in range(len(agents)):
        roll = Rollout(agents[j], deepcopy(env))
        tot_rew = 0
        for k in range(10):
            tot_rew += roll.validation_episode()
        individual_agent_rew.append(tot_rew / 10)
        state_values.append(roll.get_avg_state_value())
        agent_states.append(roll.states_encountered)



    state_values = np.array(state_values)
    state_value_var = np.var(state_values, axis=0)
    #Then rollout ensemble reward
    print("Ensemble with agents: {} and rewards: {}".format(agent_nums, individual_agent_rew))
    print("Agent avg state values: {}".format(state_values))
    print("Agent state value variance: {}".format(state_value_var))
    print("Avg cart variance: {}".format(np.mean(state_value_var[:2])))
    print("Avg pole variance: {}".format(np.mean(state_value_var[2:])))
    print("State value var averaged: {}".format(np.mean(state_value_var)))
    ensemble = Ensemble(env, agents, avfs)
    ens_roll = Rollout(ensemble, deepcopy(env))
    ens_rew = 0
    for j in range(5):
        ens_rew += ens_roll.validation_episode()
    ens_rew = ens_rew / 5
    print("Reward: {}".format(ens_rew))
    print("Agent Action Frequencies: {}".format(ensemble.get_counts()))
    print("Agent Average Risk: {}".format(ensemble.get_risks()))
    print("Average action variance among agents: {}".format(ensemble.get_act_variance()))

    ens_states = np.array(ens_roll.states_encountered)
    indices = np.arange(len(ens_states))
    if len(ens_states) > 10000:
        indices = np.random.choice(indices, size=10000, replace=False)
        ens_states = ens_states[indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Agent visited states')
    for i in range(len(agents)):
        i_states = np.array(agent_states[i]) #Array of ~200 states
        cart_pos = i_states[:, 0]
        cart_vel = i_states[:, 1]
        pole_pos = i_states[:, 2]
        pole_vel = i_states[:, 3]
        
        ax1.scatter(x=cart_pos, y=cart_vel, label='Agent {}'.format(agent_nums[i]), alpha=0.2)
        ax2.scatter(x=pole_pos, y=pole_vel, label='Agent {}'.format(agent_nums[i]), alpha=0.2)

    ens_cart_pos = ens_states[:, 0]
    ens_cart_vel = ens_states[:, 1]
    ens_pole_pos = ens_states[:, 2]
    ens_pole_vel = ens_states[:, 3]
    ax1.scatter(x=ens_cart_pos, y=ens_cart_vel, label='Ensemble', alpha=0.2)
    ax2.scatter(x=ens_pole_pos, y=ens_pole_vel, label='Ensemble', alpha=0.2)
    plt.legend()
    
    ax1.set(xlabel='Cart position', ylabel='Cart velocity')
    ax2.set(xlabel='Pole position', ylabel='Pole velocity')
    plt.savefig('./State_clusters/{}_rew_{:.2f}_var.png'.format(int(ens_rew), np.mean(state_value_var)))

    

    
    
    

    

    


