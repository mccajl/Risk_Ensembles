from risk_estimation.risk_estimation import Approximator, Trainer, Approx_Buffer
from DQN.helpers import ReplayBuffer, make_gym_env
from PolicyEnsembles.agent import Agent
from DQN.models import DQN, CnnDQN, DropoutDQN
from copy import deepcopy
import numpy as np
import argparse


def evaluate_episode(env, agent, args, N=10):
    states = []
    fails = []
    rewards = []
    for i in range(N):
        episode_fails = []
        done = False
        state = env.reset()
        tot_r = 0
        while not done:
            action, _ = agent.act(state, 0)
            states.append(state)
            next_state, reward, done, info = env.step(int(action.cpu()))
            tot_r += reward
            if info['failure'] == 1:
                episode_fails.append(1)
            else:
                episode_fails.append(0)
            
            state = next_state
        
        fail_values = list(pad_failures(episode_fails, args.H))
        fails = fails + fail_values
        rewards.append(tot_r)
    print("Mean reward: {}".format(sum(rewards)/len(rewards)))
    return states, fails

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--avf_path", type=str, default=None)
    parser.add_argument("--agent_path", type=str, default=None)
    parser.add_argument("--agent_info", type=int, default=None)
    parser.add_argument("--H", type=int, default=20)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    state_size = env.observation_space.shape[0]
    agent_info_size = 1
    network = Approximator(tuple([state_size+agent_info_size]))
    buff = Approx_Buffer()
    trainer = Trainer(network, buff)
    trainer.load_weights(args.avf_path)

    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    agent = Agent(env, q_network, target_q_network)
    agent.load_weights(args.agent_path)

    states, failures = evaluate_episode(env, agent, args)


    failures = np.reshape(failures, [-1, 1])
    #print(np.array(states).shape)
    #print(np.array(failures).shape)
    agent_info = np.zeros([len(states), agent_info_size])
    agent_info.fill(args.agent_info)
    #print(agent_info.shape)

    accuracy, average = trainer.evaluation(states, agent_info, failures)
    print("AVF Accuracy: ", accuracy)
    print("Prediction average: ", average)



