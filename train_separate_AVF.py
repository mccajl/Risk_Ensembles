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

    print("Average Reward: {}".format(np.mean(ep_reward)))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--H", type=int, default=20)
    
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    agent = Agent(env, q_network, target_q_network, 3303)

    if args.env == 'Cartpole':
        #agent.load_weights('weights/CartEnsemble1/q_net_2411.pt')
        #agent.load_weights('weights/CartEnsemble2/q_net_2050.pt')
        #agent.load_weights('weights/CartEnsemble3/q_net_3862.pt')
        agent.load_weights('weights/CartEnsemble4/q_net_2823.pt')
    else:
        #agent.load_weights('weights/MCEnsemble1/q_net_3882.pt')
        #agent.load_weights('weights/MCEnsemble2/q_net_2968.pt')
        #agent.load_weights('weights/MCEnsemble3/q_net_3303.pt')
        agent.load_weights('weights/MCEnsemble4/q_net_3563.pt')
        

    avf = Trainer(Approximator(tuple([env.observation_space.shape[0]])), Approx_Buffer(), training_iter=32,
                                use_agent_info=False)

    if args.env == 'Cartpole':
        states, agent_infos, failures = fill_buffer(agent, env, avf, 200, H=args.H) #Fill to around full replay buffer (100k)
    else:
        states, agent_infos, failures = fill_buffer(agent, env, avf, 200, H=args.H, mc=True)
    
    indices = np.arange(len(states))
    np.random.shuffle(indices)
    states = states[indices]
    agent_infos = agent_infos[indices]
    failures = failures[indices]
    train_split = int(.8*len(states))
    train_states = states[:train_split]
    train_info = agent_infos[:train_split]
    train_fail = failures[:train_split]
    val_states = states[train_split:]
    val_info = agent_infos[train_split:]
    val_fail = failures[train_split:]
    avf.add_experience(train_states, train_info, train_fail)

    num_epochs = 30
    for j in range(num_epochs): #100 epochs 
        print("Epoch {}".format(j))
        avf.train(verbose=True)
        if j < num_epochs-1:
            val_acc, _ = avf.evaluation(val_states, val_info, val_fail, print_values=False)
        else:
            val_acc, _ = avf.evaluation(val_states, val_info, val_fail, print_values=True)
        print("Validation Acc: {}".format(val_acc))
    
    if args.env == 'Cartpole':
        avf.save_weights('weights/CartEnsemble4')
    else:
        avf.save_weights('weights/MCEnsemble4')


