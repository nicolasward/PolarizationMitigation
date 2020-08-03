from matplotlib import pyplot as plt
import numpy as np

class EpsilonGreedy(object):
    
    def __init__(self, number_of_arms = 5, epsilon = 0.1, depolarize = False, lower = 0.05, upper = 0.8):
        
        self._number_of_arms = number_of_arms
        self._epsilon        = epsilon  
        self._depolarize     = depolarize
        self._lower          = lower
        self._upper          = upper
        self._name           = "Epsilon Greedy Agent"
        
        self.reset()
        
    def correct_polarization(self, greedy):
        """
        Inputs: none.
        
        Outputs:
        - recommendation in {0,...,numer_of_arms-1} corresponding to the depolarized version of an epsilon-greedy agent.
        """
        freqs = self.N / sum(self.N) if sum(self.N) > 0 else np.zeros(self._number_of_arms)

        resample = True
        count = 0

        if greedy:
            while(resample):
                count += 1
                rec = np.random.choice(np.where(self.Q == max(self.Q))[0]) if count == 1 else np.random.randint(self._number_of_arms)
                if freqs[rec] < self._upper: resample = False
            return rec
        else:
            while(resample):
                rec = np.random.randint(self._number_of_arms)
                if freqs[rec] < self._upper: resample = False
            return rec 
    
    def step(self):
        """
        Samples a new action (arm) according to an epsilon-greedy policy.
        
        Inputs:
        - depolarize : boolean value (default = False) which determines whether to correct polarization or not.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        greedy = np.random.random() > self._epsilon
             
        if self._depolarize:
            return self.correct_polarization(greedy)
        else: 
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