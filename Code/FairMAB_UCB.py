from matplotlib import pyplot as plt
import numpy as np

class FairMAB_UCB(object):

    def __init__(self, number_of_arms = 5, bonus_multiplier = 1.0, constraints = [0.1, 0.1, 0.1, 0.1, 0.1], tolerance = 0.02):
        
        self._number_of_arms = number_of_arms
        self._multiplier     = bonus_multiplier
        self._constraints    = np.asarray(constraints)
        self._alpha          = tolerance
        self._name           = "Fair UCB agent"
        
        self.reset()
   
    def step(self):
        """
        Selects a new action (arm) according to a Fair UCB policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        
        rec = -1
        x = self._constraints * (self.t - 1) - self.N
        A = np.argmax(x)
        
        if max(x) > self._alpha:
            return(A)
        else:
            return np.random.choice(np.where(self.s == max(self.s))[0])
        
               
    def update(self, previous_action, reward):
        """
        Updates UCB agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        self.t += 1.0
        self.N[previous_action] += 1
        error = reward - self.Q[previous_action]
        self.w = 1. / self.N[previous_action]
        self.Q[previous_action] += self.w * error
        
        self.U = np.sqrt(2 * np.log(self.t) / self.N)
        self.s = self.Q + self._multiplier * self.U
        
                
    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.Q = np.zeros(self._number_of_arms)
        self.N = np.ones(self._number_of_arms)
        self.t = 0.0
        self.U = 0.0
        self.s = self.Q + self._multiplier * self.U
        
        self.w = 0.0