import numpy as np
import gym
from gym import Env, spaces
from gym.utils import seeding


class ExpandedMonsterGrid(Env):

    """
    This is an implementation of the gridworld described in Contrastive Explanations for Reinforcement Learning
    in terms of Expected Consequences, van der Waa et al 2018. This version is expanded, with the board mirrored over the right y-axis.
    The board is 14x7, with the state being the x, y coordinates, and three binary features indicating presence
    of a forest, trap, or monster in the adjacent tiles.
    Each time step incurs -1 reward, being in the forest incurs -10 reward, and stepping on a trap or being adjacent to the monster incurs -100 reward.
    The agent starts in the bottom left (0, 0) and the goal tile is in (6, 6). Reaching the goal incurs +100 reward
    Forest tiles are placed in (0, 6), (0, 5), (0, 4), (1, 6), (1, 5), (2, 6), (13, 6), (13, 5), (13, 4), (12, 6), (12, 5), (11, 6)
    Traps in tiles (2, 0), (4, 0), (4, 2), (6, 2), (11, 0), (9, 0), (9, 2), (7, 2)
    Monsters start in tiles (4, 1) and (9, 1). If agent enters 'red' area, monsters move towards agent until agent leaves
    Actions:
    0: Up
    1: Right
    2: Down
    3: Left
    """
    def __init__(self, ts_max=200, seed=None):

        self.seed(seed)

        self.nA = 4

        self.shape = (14, 7)
        self.start_coords = (13, 0)
        self.coords = [self.start_coords[0], self.start_coords[1]]
        self.forest_tiles = [[0,6], [0,5], [0,4], [1,6], [1,5], [2,6],
                             [13,6], [13,5], [13,4], [12,6],[12,5], [11,6]]
        self.trap_tiles = [[2,0], [4,0], [4,2], [6,2],
                           [11,0], [9,0], [9,2], [7,2]]
        self.monster_start = [[4,1],[9,1]]
        self.monster_tile = [[4,1],[9,1]]
        self.num_monsters = len(self.monster_tile)
        self.terminal_tile = [6,6]
        self.observation_space = spaces.MultiDiscrete([14, 7, 2, 2, 2])
        self.action_space = spaces.Discrete(self.nA)

        self.state = self.form_state()
        self.act_dict = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}

        self.ts = 0
        self.max_ts = ts_max
        self._form_red_area()

    
    def _form_red_area(self):
        self.red_area = {} #Form red area on map according to paper
        for i in range(14):
            for j in range(7):
                if i == 4 and j == 4:
                    self.red_area[(i, j)] = True
                elif i == 2 and j == 0:
                    self.red_area[(i, j)] = True
                elif i == 11 and j == 0:
                    self.red_area[(i, j)] = True
                elif i == 9 and j ==4:
                    self.red_area[(i, j)] = True
                elif i >= 3 and i <= 10 and j <= 3:
                    self.red_area[(i, j)] = True
                else:
                    self.red_area[(i, j)] = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_adjacent_info(self):
        adj_info = [self.adj_forest(), self.adj_trap(), self.adj_monster()]
        return adj_info

    def adj_forest(self):
        x = self.coords[0]
        y = self.coords[1]
        for f in self.forest_tiles:
            fx = f[0]
            fy = f[1]
            if abs(x-fx) + abs(y-fy) == 1:
                return 1
        return 0

    def adj_trap(self):
        x = self.coords[0]
        y = self.coords[1]
        for t in self.trap_tiles:
            tx = t[0]
            ty = t[1]
            if abs(x-tx) + abs(y-ty) == 1:
                return 1
        return 0

    def adj_monster(self):
        x = self.coords[0]
        y = self.coords[1]
        for m in self.monster_tile:
            mx = m[0]
            my = m[1]
            if abs(x-mx) + abs(y-my) == 1:
                return 1
        return 0 
    
    def _check_monster_boundaries(self, action, monster_idx):
        new_x = self.monster_tile[monster_idx][0] + self.act_dict[action][0]
        new_y = self.monster_tile[monster_idx][1] + self.act_dict[action][1]

        if new_y < 0: #Possible monster out-of-bounds position(s)
            return 0
        
        if self.red_area[(new_x, new_y)]:
            return 1
        else:
            return 0
    
    def _monster_move_values(self, action, monster_idx):
        new_x = self.monster_tile[monster_idx][0] + self.act_dict[action][0]
        new_y = self.monster_tile[monster_idx][1] + self.act_dict[action][1]
        dist = abs(new_x - self.coords[0]) + abs(new_y - self.coords[1])
        return dist

    def move_monster(self, monster_idx):
        cur_x = self.monster_tile[monster_idx][0]
        cur_y = self.monster_tile[monster_idx][1]
        possible_actions = [0, 1, 2, 3]
        legal_actions = []
        for a in possible_actions:
            legal = self._check_monster_boundaries(a, monster_idx)
            if legal:
                legal_actions.append(a)
        
        dists = []
        for a in legal_actions:
            dists.append(self._monster_move_values(a, monster_idx))
        best_action = legal_actions[np.argmin(dists)]
        self.monster_tile[monster_idx][0] = cur_x + self.act_dict[best_action][0]
        self.monster_tile[monster_idx][1] = cur_y + self.act_dict[best_action][1]
            
    def in_red_area(self):
        pos = (self.coords[0], self.coords[1])
        return self.red_area[pos]

    def in_forest(self):
        x = self.coords[0]
        y = self.coords[1]
        for f in self.forest_tiles:
            fx = f[0]
            fy = f[1]
            if x == fx and y == fy:
                return 1
        return 0
    
    def in_trap(self):
        x = self.coords[0]
        y = self.coords[1]
        for t in self.trap_tiles:
            tx = t[0]
            ty = t[1]
            if tx == x and ty == y:
                return 1
        
        return 0
    
    def at_goal(self):
        if self.coords[0] == self.terminal_tile[0] and self.coords[1] == self.terminal_tile[1]:
            return 1
        else:
            return 0

    def _limit_coordinates(self, action):
        pos = self.coords
        x = min(self.coords[0] + self.act_dict[int(action)][0], self.shape[0]-1)
        x = max(x, 0)
        y = min(self.coords[1] + self.act_dict[int(action)][1], self.shape[1]-1)
        y = max(y, 0)
        return x, y
    
    def get_reward(self):
        if self.adj_monster():
            return -100
        elif self.in_trap():
            return -100
        elif self.in_forest():
            return -5
        elif self.at_goal():
            return 100
        else:
            return -1

    def form_state(self):
        state = self.coords + self.get_adjacent_info()
        return state

    def step(self, action):
        new_x, new_y = self._limit_coordinates(action)
        self.coords[0] = new_x
        self.coords[1] = new_y
        rew = self.get_reward()
        info = {}
        if self.at_goal():
            done = True
            info['failure'] = 0
        elif self.ts > self.max_ts:
            done = True
            info['failure'] = 1
        else:
            done = bool(self.adj_monster() or self.in_trap())
            info['failure'] = max(self.adj_monster(), self.in_trap())
        
        if self.in_red_area(): #Move monster after agent has had a chance to move. If agent is still adjacent, then monster has caught agent.
            for i in range(self.num_monsters):
                self.move_monster(i)
        
        self.state = self.form_state()
        self.ts += 1
        return self.state, rew, done, info



    def reset_state(self, start_coords, monster_tile=None):
        #Additional reset function which allows for rollouts of episodes starting from a specified start state
        #Monster_tile is optional coordinates to start each monster in (2d array)
        self.coords[0] = start_coords[0]
        self.coords[1] = start_coords[1]
        if monster_tile is not None:
            for i in range(len(monster_tile)):
                self.monster_tile[i][0] = monster_tile[i][0]
                self.monster_tile[i][1] = monster_tile[i][1]
        else:
            self.monster_tile[0] = [4,1]
            self.monster_tile[1] = [9,1]
        
        self.state = self.form_state()
        self.ts = 0
        return self.state
    
    def reset(self):
        self.coords = [self.start_coords[0],self.start_coords[1]]
        self.monster_tile[0] = [4,1]
        self.monster_tile[1] = [9,1]
        self.state = self.form_state()
        self.ts = 0
        return self.state
    
    def all_coords(self):
        coord_set = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                coord_set.append((i, j))
        return coord_set

    def parse_state(self, state): #An environment function for returning the state values important for risk estimation
        return [state[0], state[1]]
