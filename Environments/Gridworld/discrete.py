import numpy as np
import gym
from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class OneHotEncoding(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (size,), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size

class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.num_steps = 0

        self.action_space = spaces.Discrete(self.nA)
        #self.observation_space = spaces.Discrete(self.nS)
        self.observation_space = OneHotEncoding(int(self.nS))

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_state(self, start_state):
        #Additional reset function which allows for rollouts of episodes starting from a specified start state
        self.s = start_state
        self.lastaction = None
        self.state = np.zeros(self.nS)
        self.state[int(self.s)] = 1
        self.num_steps = 0
        return self.state
    
    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.state = np.zeros(self.nS)
        self.state[int(self.s)] = 1
        self.num_steps = 0
        return self.state

    def step(self, a):
        self.num_steps += 1
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, failed = transitions[i]
        self.s = s
        self.lastaction = a
        self.state = np.zeros(self.nS)
        self.state[self.s] = 1
        if self.num_steps >= 1000:
            d = True
            failed = 1
        return (self.state, r, d, {"prob": p, "failure": failed})