from PolicyEnsembles.agent import Agent
from risk_estimation.risk_estimation import Rollout, Trainer, Approximator, Approx_Buffer
from PolicyEnsembles.ensemble import Ensemble
import argparse
from DQN.helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from DQN.models import DQN, CnnDQN, DropoutDQN
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    env = make_gym_env(args.env)
    q_network = DQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    if args.env == 'Cartpole':
        agent1 = Agent(env, q_network, target_q_network, 2411)
        agent1.load_weights('weights/CartEnsemble1/q_net_2411.pt')
        agent2 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 2050)
        agent2.load_weights('weights/CartEnsemble2/q_net_2050.pt')
        agent3 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 3862)
        agent3.load_weights('weights/CartEnsemble3/q_net_3862.pt')
        agent4 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 2823)
        agent4.load_weights('weights/CartEnsemble4/q_net_2823.pt')
    else:
        agent1 = Agent(env, q_network, target_q_network, 3882)
        agent1.load_weights('weights/MCEnsemble1/q_net_3882.pt')
        agent2 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 2968)
        agent2.load_weights('weights/MCEnsemble2/q_net_2968.pt')
        agent3 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 3303)
        agent3.load_weights('weights/MCEnsemble3/q_net_3303.pt')
        agent4 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 3563)
        agent4.load_weights('weights/MCEnsemble4/q_net_3563.pt')
    
    agents = [agent1, agent2, agent3, agent4]
    avf1 = Trainer(Approximator(tuple([env.observation_space.shape[0]])), Approx_Buffer(), use_agent_info=False)
    #avf1 = Trainer(Approximator(tuple([env.observation_space.shape[0]+1])), Approx_Buffer(), use_agent_info=True)
    avf2 = deepcopy(avf1)
    avf3 = deepcopy(avf1)
    avf4 = deepcopy(avf1)
    if args.env == 'Cartpole':
        avf1.load_weights('weights/CartEnsemble1')
        avf2.load_weights('weights/CartEnsemble2')
        avf3.load_weights('weights/CartEnsemble3')
        avf4.load_weights('weights/CartEnsemble4')
    else:
        avf1.load_weights('weights/MCEnsemble1')
        avf2.load_weights('weights/MCEnsemble2')
        avf3.load_weights('weights/MCEnsemble3')
        avf4.load_weights('weights/MCEnsemble4')

    avfs = [avf1, avf2, avf3, avf4]

    r1 = Rollout(agent1, env)
    r2 = Rollout(agent2, env)
    r3 = Rollout(agent3, env)
    r4 = Rollout(agent4, env)
    rollouts = [r1, r2, r3, r4]


    for i, r in enumerate(rollouts):
        tot_rew = 0
        for j in range(10):
            rew = r.validation_episode()
            tot_rew += rew
        
        print("Agent {}: Reward {}".format(i+1, tot_rew/10))
        
    ensemble = Ensemble(env, agents, avfs, use_agent_info=False)
    ensemble_rollout = Rollout(ensemble, env)
    
    tot_rew = 0
    for i in range(50):
        rew = ensemble_rollout.validation_episode()
        tot_rew += rew
    
    print("Ensemble Agent: Reward {}".format(tot_rew/50))
    print("Agent Action Frequencies: {}".format(ensemble.get_counts()))
    print("Agent Average Risk: {}".format(ensemble.get_risks()))

