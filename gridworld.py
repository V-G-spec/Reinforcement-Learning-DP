import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
# UP2 = 4
# RIGHT2 = 5
# DOWN2 = 6
# LEFT2 = 7

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """
        
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[10, 10]): # Default shape: 10, 10
        #####
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) # 100
        nA = 4 #Up, down, left, right

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])
        init_pos = 10
        
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == nS-2 # Last row, 2nd last column. So nS-2 because s starts with 0
            is_mine = lambda s: s in (46, 66, 84)
            
            reward = 0 if is_done(s) else -1

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                # P[s][UP2] = [(3.0, s, reward, True)]
                # P[s][RIGHT2] = [(1.0, s, reward, True)]
                # P[s][DOWN2] = [(1.0, s, reward, True)]
                # P[s][LEFT2] = [(1.0, s, reward, True)]
            
            # We land on a mine
            elif is_mine(s):
                P[s][UP] = [(1.0, init_pos, reward, False)]
                P[s][LEFT] = [(1.0, init_pos, reward, False)]
                P[s][DOWN] = [(1.0, init_pos, reward, False)]
                P[s][RIGHT] = [(1.0, init_pos, reward, False)]
                # P[s][UP2] = [(1.0, init_pos, reward, False)]
                # P[s][LEFT2] = [(1.0, init_pos, reward, False)]
                # P[s][DOWN2] = [(1.0, init_pos, reward, False)]
                # P[s][RIGHT2] = [(1.0, init_pos, reward, False)]
                
                
                
            # Not a terminal state nor it is a mine
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                
                if (y==1):
                    ns_up2 = s-MAX_X
                elif (y==0):
                    ns_up2 = s
                else:
                    ns_up2 = s - 2*MAX_X
                    
                if (y==MAX_X-2):
                    ns_right2 = s + 1
                elif (y==MAX_X-1):
                    ns_right2 = s
                else:
                    ns_right2 = s + 2
                
                if (y==MAX_Y-2):
                    ns_down2 = s+MAX_X
                elif (y==MAX_Y-1):
                    ns_down2 = s
                else:
                    ns_down2 = s + 2*MAX_X
                    
                if (x==1):
                    ns_left2 = s-1
                elif (x==0):
                    ns_left2 = s
                else:
                    ns_left2 = s - 2
                
                P[s][UP] = [(0.14, ns_up, reward, is_done(ns_up)), (0.86, ns_up2, reward, is_done(ns_up2))]
                P[s][RIGHT] = [(0.38, ns_right, reward, is_done(ns_right)), (0.62, ns_right2, reward, is_done(ns_right2))]
                P[s][DOWN] = [(0.14, ns_down, reward, is_done(ns_down)), (0.86, ns_down2, reward, is_done(ns_down2))]
                P[s][LEFT] = [(0.38, ns_left, reward, is_done(ns_left)), (0.62, ns_left2, reward, is_done(ns_left2))]
                

            it.iternext()

        # Initial state distribution is constant
        isd = np.zeros((nS))
        isd[init_pos] = 1

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)



        #####

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            if self.s == s:
                output = " x "
            elif s == self.nS - 2:
                output = " T "
            elif s in (46, 66, 84):
                output = " M "
            else:
                output = " o "
                
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

            
def main():
    env = GridworldEnv()
    env._render()
    for k in env.P.keys():
        print(f'env.P[{k}]={env.P[k]}')
    
if __name__ == "__main__":
    main()
