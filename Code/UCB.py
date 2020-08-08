from matplotlib import pyplot as plt
import numpy as np

class UCB(object):

    def __init__(self, number_of_arms = 5, bonus_multiplier = 1.0, depolarize = False, lower = 0.05, upper = 0.8):
        
        self._number_of_arms = number_of_arms
        self._multiplier     = bonus_multiplier
        self._depolarize     = depolarize
        self._lower          = lower
        self._upper          = upper
        self._name           = "UCB Agent"
        
        self.reset()
        
        
    def correct_polarization(self):
        """
        Inputs: none.
        
        Outputs:
        - recommendation in {0,...,numer_of_arms-1} corresponding to the depolarized version of a UCB agent.
        """
        freqs = (self.N - 1) / (sum(self.N) - self._number_of_arms) if sum(self.N) - self._number_of_arms > 0 else np.zeros(self._number_of_arms)

        resample = True
        count    = 0

        while(resample):
            count += 1
            rec = np.random.choice(np.where(self.s == max(self.s))[0]) if count == 1 else np.random.randint(self._number_of_arms)
            if freqs[rec] < self._upper: resample = False
        return rec
    
 
    def step(self):
        """
        Selects a new action (arm) according to a UCB policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        
        if self._depolarize:
            return self.correct_polarization()
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
        self.alpha = 1. / self.N[previous_action]
        self.Q[previous_action] += self.alpha * error
        
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
        
        self.alpha = 0.0