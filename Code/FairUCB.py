import numpy as np

class FairUCB(object):

    def __init__(self, number_of_arms = 5, bonus_multiplier = 1.0, lower = 0.05, upper = 1):
        
        self._number_of_arms = number_of_arms
        self._multiplier     = bonus_multiplier
        self._lower          = lower
        self._upper          = upper 
        self._name           = "Fair UCB Agent"
        
        self.reset()
 
    def step(self):
        """
        Selects a new action (arm) according to a UCB policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        freqs = (self.N - 1) / (sum(self.N) - self._number_of_arms) if sum(self.N) - self._number_of_arms > 0 else np.zeros(self._number_of_arms)

        resample = True
        count = 0

        while(resample):
            count += 1
            rec = np.random.choice(np.where(self.target == max(self.target))[0]) if count == 1 else np.random.randint(self._number_of_arms)
            if freqs[rec] < self._upper: resample = False
        return rec
        
    def update(self, previous_action, reward):
        """
        Updates UCB agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        # Increment relevant parameters
        self.t += 1.0
        self.N[previous_action] += 1
        self.played_at_t[previous_action] += 1
        
        # Update value for previous_action
        error      = reward - self.Q[previous_action]
        self.alpha = 1. / self.N[previous_action]
        self.Q[previous_action] += self.alpha * error
        
        # Update queue
        self.C += self._lower - self.played_at_t
        self.C  = np.asarray([0 if i < 0 else i for i in self.C])
        
        # Set new target to maximize
        self.U = np.sqrt(2 * np.log(self.t) / self.N)
        self.target = self.C + self._multiplier * (self.Q + self.U)
        
        # Reset next round's play counts
        self.played_at_t = np.zeros(self._number_of_arms)        
        
    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.Q      = np.zeros(self._number_of_arms)
        self.C      = np.zeros(self._number_of_arms)
        self.U      = np.zeros(self._number_of_arms)
        self.target = np.zeros(self._number_of_arms)
        
        self.N = np.ones(self._number_of_arms)
        self.t = 0.0
        self.alpha = 0.0
        self.played_at_t = np.zeros(self._number_of_arms)