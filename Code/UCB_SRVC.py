import numpy as np

class UCB_SRVC(object):

    def __init__(self, number_of_arms = 5, multiplier = 1, window = 20, function = 1):
        
        self._number_of_arms = number_of_arms
        self._multiplier     = multiplier
        self._window         = window
        self._name           = "UCB-SRVC Agent"
        self._function       = function
        self.reset()
     
    def step(self):
        """
        Selects a new action (arm) according to a UCB policy.
        
        Inputs: none.
        
        Outputs: 
        - integer in {0,...,number_of_arms-1} corresponding to the next action to take.
        """
        return np.random.choice(np.where(self.D == max(self.D))[0])     
               
    def update(self, previous_action, reward):
        """
        Updates constrained UCB agent parameters.
        
        Inputs:
        - previous_action : integer in [0, number_of_arms-1] corresponding to the previous action taken.
        - reward          : integer in {0, 1} corresponding to the reward yielded by previous_action.
        
        Outputs: none.
        """
        self.t += 1.0
        self.N[previous_action] += 1
        
        if self.t % self._window != 0:
            if self._function == 2:
                self.costs[previous_action] += self._multiplier * self.N[previous_action]
            elif self._function == 3:
                self.costs[previous_action] += self._multiplier * (self.N[previous_action] ** 2)
            else:
                self.costs[previous_action] += self._multiplier * np.log(self.N[previous_action])
        else:
            self.costs = np.ones(self._number_of_arms)
            
        self.l = np.min(self.costs)
        
        error = reward - self.Q[previous_action]
        self.alpha = 1. / self.N[previous_action]
        self.Q[previous_action] += self.alpha * error
        
        self.U = np.sqrt(np.log(self.t) / self.N)
        self.D = (self.Q / self.costs) + (1.0 / self.l) * (1.0 + 1.0 / (self.l - self.U)) * self.U

    def reset(self):
        """
        Reset agent parameters to initial values.
        """
        self.Q = np.zeros(self._number_of_arms)
        self.N = np.ones(self._number_of_arms)
        self.costs = 1 * np.ones(self._number_of_arms)
        self.l = np.min(self.costs)
        self.t = 0.0
        self.U = 0.0
        self.D = self.Q / self.costs
        
        self.alpha = 0.0