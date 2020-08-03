from matplotlib import pyplot as plt
import numpy as np

def train(data, agent, steps = 1000):
    """
    Trains a bandit agent on provided data for a given number of steps.
    
    Inputs:
    - data  : (n x 3)-dimensional dataset.
    - agent : bandit agent (one of EpsilonGreedy, UCB, or Thompson).
    - steps : integer indicating the number of steps to train the agent.
    
    Outputs:
    - history : history of recommendations selected during training.
    """
    history = np.array([0, 0, 0])
    
    for i in range(steps):
        
        recommendation = agent.step() + 1
        
        if recommendation in data[:, 1]:
            
            # Set aside all rows corresponding to this recommendation
            matches  = data[data[:, 1] == recommendation]
            
            # Select one of these rows randomly
            instance = matches[np.random.choice(matches.shape[0])]
            
            # Save the row to history
            history  = np.vstack((history, instance))
            
            # Update agent's model of the environment
            agent.update(recommendation-1, instance[2])
            
    print("----- Successfully trained", agent._name, "-----")
            
    return history[1:]

def score(history, rolling = 50, n_arms = 5):
    """
    Plots cumulative reward, mean reward and rolling mean reward for a given training history.
    
    Inputs:
    - history : (h x 3)-dimensional array of samples seen during training.
    - rolling : number of steps to average across when computing moving average.
    
    Outputs: none.
    """
    timesteps         = [i+1 for i in range(history.shape[0])]
    cumulative_reward = np.cumsum(history[:, 2])
    mean_reward       = cumulative_reward / np.asarray(timesteps)
    rolling_mean      = [np.mean(history[i-rolling:i, 2]) for i in range(rolling, history.shape[0])]
    counts            = np.asarray([history[history[:, 1] == i+1, ].shape[0] for i in range(n_arms)]) / history.shape[0]
    cumulative_regret = np.cumsum(1 - history[:, 2])
    mean_regret       = cumulative_regret / np.asarray(timesteps)
    max_gap           = (counts[np.where(counts == max(counts))] - counts[np.where(counts == min(counts))])[0]
    
    breakdown = []
    for j in range(n_arms):
        b = np.cumsum(history[:, 1] == j+1)
        t = np.asarray(timesteps)
        breakdown.append(b / t)
        
    breakdown_rolling = []
    for b in breakdown:
        breakdown_rolling.append([np.mean(b[i-rolling:i]) for i in range(rolling, len(b))])
    
    figure, _ = plt.subplots(4, 1)
    figure.tight_layout(pad = 7.0)
    
    plt.subplot(6, 1, 1)
    plt.plot(timesteps, cumulative_reward, '-')
    plt.title('Cumulative reward over time')
    plt.xlabel('Time')
    plt.ylabel('Cumul. reward')

    plt.subplot(6, 1, 2)
    plt.plot(timesteps, mean_reward, '-')
    plt.title('Mean reward over time')
    plt.xlabel('Time')
    plt.ylabel('Mean reward')
    
    plt.subplot(6, 1, 3)
    plt.plot([i+1 for i in range(len(rolling_mean))], rolling_mean, '-')
    plt.title('%i-step rolling mean reward' % rolling)
    plt.xlabel('Time')
    plt.ylabel('Rolling mean reward (%i steps)' % rolling)
    
    plt.subplot(6, 1, 4)
    plt.plot(timesteps, mean_regret, '-')
    plt.title('Mean regret over time')
    plt.xlabel('Time')
    plt.ylabel('Mean regret')
    
    plt.subplot(6, 1, 5)
    for j in range(n_arms):
        l  = "Group " + str(j+1)
        br = breakdown_rolling[j]
        plt.plot([i+1 for i in range(len(br))], br, label = l)
    plt.legend()
    plt.title('%i-step rolling recommendation proportion over time' % rolling)
    plt.xlabel('Time')
    plt.ylabel('Rolling rec. frac. (%i steps)' % rolling)
    
    
    fig = plt.subplot(6, 1, 6)
    groups = [i+1 for i in range(n_arms)]
    fig.set_xticks(groups)
    plt.bar(groups, counts)
    plt.title('Frequency of each group in recommendation history')
    plt.xlabel('Group')
    plt.ylabel('Frequency')
    
    print("Group frequencies                 :", counts)
    print("Max gap between group frequencies :", max_gap)

    plt.show()
    
def entropy(values):
    """
    Computes entropy of a list of values.
    
    Inputs:
    - values : list or 1D array of values (probabilities or not).
    
    Outputs:
    - entropy of these values.
    """
    if sum(values) == 1.0:
        return sum(np.asarray(values) * np.log(values))
    else:
        softmax = np.exp(values) / sum(np.exp(values))
        return sum(softmax * np.log(softmax))