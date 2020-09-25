from matplotlib import pyplot as plt
import numpy as np

class FairEG(object):
    
    def __init__(self, number_of_arms = 5, epsilon = 0.1, constraints = [0.1, 0.1, 0.1, 0.1, 0.1], tolerance = 0.02):
        
        self._number_of_arms = number_of_arms
        self._epsilon        = epsilon  
        self._constraints    = np.asarray(constraints)
        self._alpha          = tolerance
        self._name           = "Fair Epsilon Greedy agent"
        
        self.reset()

    
    def step(self):
        """
        Samples a new action (arm) according to a fair epsilon-greedy policy.
        
        Inputs:
        - depolarize : boolean value (default = False) which determines whether to correct polarization or not.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        self.t += 1.0
        rec = -1
        s = self._constraints * (self.t - 1) - self.N
        A = np.argmax(s)
        
        if max(s) > self._alpha:
            return(A)
        else:
            greedy = np.random.random() > self._epsilon
            if greedy:
                # exploit action with highest expected value (with random tie break)
                return np.random.choice(np.where(self.Q == max(self.Q))[0])
            else:
                # explore other action
                return np.random.randint(self._number_of_arms)                  
                        
    def update(self, previous_action, reward):
        """
        Updates epsilon-greedy agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        self.N[previous_action] += 1
        
        alpha = 1. / self.N[previous_action]
        error = reward - self.Q[previous_action]
        
        self.Q[previous_action] += alpha * error
        
    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.Q = np.zeros(self._number_of_arms)
        self.N = np.zeros(self._number_of_arms)
        self.t = 0.0