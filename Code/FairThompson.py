from matplotlib import pyplot as plt
import numpy as np

class FairThompson(object):

    def __init__(self, number_of_arms = 5, constraints = [0.1, 0.1, 0.1, 0.1, 0.1], tolerance = 0.02):
        
        self._number_of_arms = number_of_arms
        self._constraints    = np.asarray(constraints)
        self._alpha          = tolerance
        self._name           = "Fair Thompson Sampling agent"
        
        self.reset()
        
        
    def step(self):
        """
        Selects a new action (arm) according to a Fair Thompson sampling policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1] corresponding to the next action to take.
        """
        self.t += 1.0
        x = self._constraints * (self.t - 1) - self.N
        A = np.argmax(x)
        
        if max(x) > self._alpha:
            return(A)
        else:
            return np.random.choice(np.where(self.theta == max(self.theta))[0])
    
    def update(self, previous_action, reward):
        """
        Updates agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        self.N[previous_action] += 1
        self.theta = np.random.beta(self.s, self.f)
        self.s[previous_action] += reward
        self.f[previous_action] += 1 - reward
        self.play_counts[previous_action] += 1

    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.s = np.ones(self._number_of_arms)
        self.f = np.ones(self._number_of_arms)
        self.N = np.zeros(self._number_of_arms)
        self.t = 0.0
        self.theta = np.random.beta(self.s, self.f)        
        self.play_counts = np.zeros(self._number_of_arms)