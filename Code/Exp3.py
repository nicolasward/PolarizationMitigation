from matplotlib import pyplot as plt
import numpy as np

class Exp3(object):

    def __init__(self, number_of_arms = 5, gamma = 0.1, depolarize = False, lower = 0.05, upper = 0.8):
        
        self._number_of_arms = number_of_arms
        self._gamma          = gamma
        self._depolarize     = depolarize
        self._lower          = lower
        self._upper          = upper
        self._name           = "Exp3 Agent"
        
        self.reset()
        
    def correct_polarization(self):
        """
        Inputs: none.
        
        Outputs:
        - recommendation in {0,...,numer_of_arms-1} corresponding to the depolarized version of an Exp3 agent.
        """
        freqs = self.play_counts / sum(self.play_counts) if sum(self.play_counts) > 0 else np.zeros(self._number_of_arms)

        resample = True
        count = 0

        while(resample):
            count += 1
            rec = np.random.choice(range(self._number_of_arms), p = self.p) if count == 1 else np.random.randint(self._number_of_arms)
            if freqs[rec] < self._upper: resample = False
        return rec
        
    def step(self):
        """
        Selects a new action (arm) according to an Exp3 policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        if self._depolarize:
            return self.correct_polarization()
        else:
            return np.random.choice(range(self._number_of_arms), p = self.p)        
    
    def update(self, previous_action, reward):
        """
        Updates Exp3 agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        estimated_reward = reward / self.p[previous_action]
        self.w[previous_action] = self.w[previous_action] * np.exp(self._gamma * estimated_reward / self._number_of_arms)
        self.p = (1 - self._gamma) * (self.w / sum(self.w)) + (self._gamma / self._number_of_arms)
        self.play_counts[previous_action] += 1
        
    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.w = np.ones(self._number_of_arms)
        self.p = np.ones(self._number_of_arms) / self._number_of_arms
        self.play_counts = np.zeros(self._number_of_arms)