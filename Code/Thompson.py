from matplotlib import pyplot as plt
import numpy as np

class Thompson(object):

    def __init__(self, number_of_arms = 5, depolarize = False, lower = 0.05, upper = 0.8):
        
        self._number_of_arms = number_of_arms
        self._depolarize     = depolarize
        self._lower          = lower
        self._upper          = upper
        self._name           = "Thompson Sampling Agent"
        
        self.reset()
        
    def correct_polarization(self):
        """
        Inputs: none.
        
        Outputs:
        - recommendation in {0,...,numer_of_arms-1} corresponding to the depolarized version of a Thompson Sampling agent.
        """
        freqs = self.play_counts / sum(self.play_counts) if sum(self.play_counts) > 0 else np.zeros(self._number_of_arms)

        resample = True
        count = 0

        while(resample):
            count += 1
            rec = np.random.choice(np.where(self.theta == max(self.theta))[0]) if count == 1 else np.random.randint(self._number_of_arms)
            if freqs[rec] < self._upper: resample = False
        return rec
        
    def step(self):
        """
        Selects a new action (arm) according to a Thompson sampling policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1] corresponding to the next action to take.
        """
        
        if self._depolarize:
            return self.correct_polarization()
        else:
            return np.random.choice(np.where(self.theta == max(self.theta))[0])
    
    def update(self, previous_action, reward):
        """
        Updates Thompson sampling agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
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
        self.theta = np.random.beta(self.s, self.f)        
        self.play_counts = np.zeros(self._number_of_arms)