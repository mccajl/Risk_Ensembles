from PolicyEnsembles.agent import Agent
from risk_estimation.risk_estimation import Rollout, Trainer, Approximator, Approx_Buffer
from PolicyEnsembles.ensemble import Ensemble
import argparse
from DQN.helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from DQN.models import DQN, CnnDQN, DropoutDQN
from copy import deepcopy


def roll_single_agent(env, agent):
    rollout = Rollout(agent, env)
    tot_rew = 0
    for j in range(50):
        rew = rollout.validation_episode()
        tot_rew += rew
    return tot_rew / 50

def roll_ensemble(env, agents, avfs):
    assert len(agents) == len(avfs)
    rollouts = []
    for i in range(len(agents)):
        rollouts.append(Rollout(agents[i], env))


    for i, r in enumerate(rollouts):
        tot_rew = 0
        for j in range(20):
            rew = r.validation_episode()
            tot_rew += rew
        
        print("Agent {}: Reward {}".format(i+1, tot_rew/20))
        
    ensemble = Ensemble(env, agents, avfs, use_agent_info=False)
    ensemble_rollout = Rollout(ensemble, env)
    
    tot_rew = 0
    for i in range(50):
        rew = ensemble_rollout.validation_episode()
        tot_rew += rew
    
    print("Ensemble Agent: Reward {}".format(tot_rew/50))
    print("Agent Action Frequencies: {}".format(ensemble.get_counts()))
    print("Agent Average Risk: {}".format(ensemble.get_risks()))

    return tot_rew/50


