from itertools import filterfalse
import numpy as np
import torch
import random


class Ensemble(object):
    def __init__(self, env, agents, risk_estimators):
        self.env = env
        self.agents = agents
        self.risk_estimators = risk_estimators
        self.agent_counts = np.zeros(len(agents))
        self.risk_averages = np.zeros(len(agents))
        self.action_variance = 0
        self.num_iters = 0

    
    def act(self, state, epsilon=0):
        actions = []
        risks = []
        for i in range(len(self.agents)):
            act, _ = self.agents[i].act(state, epsilon=epsilon)
            actions.append(act) 
            risk = self.risk_estimators[i].eval_state(self.env.parse_state(state))
            risks.append(risk)
            self.risk_averages[i] += risk
        self.num_iters += 1
        self.action_variance += np.var(np.array(actions))
        best_action, picked_agent = self._majority_vote(actions)
        #best_action, picked_agent = self._distribution_vote(state, actions)
        #best_action, picked_agent = self._return_voting(risks, actions)
        #best_action, picked_agent = self._risk_voting(risks, actions)
        self.agent_counts[picked_agent] += 1
        return best_action, 0

    def _return_voting(self, returns, actions):
        max_indices = np.argwhere(returns == np.max(returns))
        max_indices = max_indices.flatten().tolist()
        choice = np.random.randint(len(max_indices))
        best_action = np.array(actions)[max_indices[choice]]
        return best_action, max_indices[choice]

    def _risk_voting(self, risks, actions):
        min_indices = np.argwhere(risks == np.min(risks))
        min_indices = min_indices.flatten().tolist()
        choice = np.random.randint(len(min_indices))
        best_action = np.array(actions)[min_indices[choice]]
        return best_action, min_indices[choice]
    
    def _majority_vote(self, actions):
        action_counts = np.zeros(self.env.action_space.n)
        for a in actions:
            action_counts[a] += 1
        voted_actions = np.argwhere(action_counts == np.max(action_counts))
        voted_actions = voted_actions.flatten().tolist()
        choice = np.random.randint(len(voted_actions))
        best_action = np.array(actions)[voted_actions[choice]]
        return best_action, voted_actions[choice]

    def _distribution_vote(self, state, actions):
        #Vote based on which state is closest to agent's observed on-policy distribution
        least_std = float('inf')
        for i in range(len(self.agents)):
            mean_i, var_i = self.agents[i].get_state_distribution()
            mean_i = np.array(mean_i)
            var_i = np.array(var_i)
            diff = np.absolute(mean_i - np.array(state))
            std = diff / np.sqrt(var_i)
            mean_std = np.mean(std)
            if mean_std < least_std:
                least_std = mean_std
                best_agent = i
        
        return actions[i], i


    def get_counts(self):
        return self.agent_counts / np.sum(self.agent_counts)
    
    def get_risks(self):
        return self.risk_averages / self.num_iters
    
    def get_act_variance(self):
        return self.action_variance / self.num_iters
    