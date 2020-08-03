from matplotlib import pyplot as plt
import numpy as np

class RandomBandit(object):
    
    def __init__(self, n_arms = 5, depolarize = False, lower = 0.05, upper = 0.8):
        
        self._number_of_arms = n_arms
        self._name           = "Random Agent"
        
    def step(self):
        """
        Samples a new action (arm) randomly.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        return np.random.randint(self._number_of_arms)
    
    def update(self, previous_action, reward):
        pass