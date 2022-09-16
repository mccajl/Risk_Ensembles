from PolicyEnsembles.agent import Agent
from risk_estimation.failure_search import Trainer, Approximator, Approx_Buffer
from risk_estimation.risk_estimation import Rollout, EstimateRisk
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
    agent1 = Agent(env, q_network, target_q_network, 6537)
    agent1.load_weights('weights/CartpoleRisk2/q_net_6537.pt')
    agent2 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 3825)
    agent2.load_weights('weights/CartpoleRisk2/q_net_3825.pt')
    agent3 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 5286)
    agent3.load_weights('weights/CartpoleRisk2/q_net_5286.pt')
    agent4 = Agent(env, deepcopy(q_network), deepcopy(target_q_network), 2001)
    agent4.load_weights('weights/CartpoleRisk2/q_net_2001.pt')
    F_trainer = Trainer(Approximator(tuple([env.observation_space.shape[0]+1])), Approx_Buffer())
    F_trainer.load_weights('weights/CartpoleRisk2')
    estimators = []
    r1 = Rollout(agent1, env)
    r2 = Rollout(agent2, env)
    r3 = Rollout(agent3, env)
    r4 = Rollout(agent4, env)
    rollouts = [r1, r2, r3, r4]
    avfs = [F_trainer, deepcopy(F_trainer), deepcopy(F_trainer), deepcopy(F_trainer)]
    agents = [agent1, agent2, agent3, agent4]

    for i, r in enumerate(rollouts):
        tot_rew = 0
        for j in range(10):
            rew = r.validation_episode()
            tot_rew += rew
        
        print("Agent {}: Reward {}".format(i+1, tot_rew/10))
        
    ensemble = Ensemble(env, agents, avfs)
    ensemble_rollout = Rollout(ensemble, env)
    
    tot_rew = 0
    for i in range(10):
        rew = ensemble_rollout.validation_episode()
        tot_rew += rew
    
    print("Ensemble Agent: Reward {}".format(tot_rew/10))
    print("Agent Action Frequencies: {}".format(ensemble.get_counts()))
    print("Agent Average Risk: {}".format(ensemble.get_risks()))

