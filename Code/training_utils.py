from matplotlib import pyplot as plt
import numpy as np

def train(data, agent, steps = 1000, mode = 'simulated'):
    """
    Trains a bandit agent on provided data for a given number of steps.
    
    Inputs:
    - data  : (n x 3)-dimensional dataset.
    - agent : bandit agent (one of EpsilonGreedy, UCB, or Thompson).
    - steps : integer indicating the number of steps to train the agent.
    - mode  : can be either 'simulated' or 'real' and specifies whether to train on simulated data or the MovieLens dataset.
    
    Outputs:
    - history : history of recommendations selected during training.
    """
    n = data.shape[1]
    
    history = np.zeros((1, n))

    for i in range(steps):

        recommendation = agent.step() + 1

        if recommendation in data[:, -2]:

            # Set aside all rows corresponding to this recommendation
            matches  = data[data[:, -2] == recommendation]

            # Select one of these rows randomly
            instance = matches[np.random.choice(matches.shape[0])]

            # Save the row to history
            history  = np.vstack((history, instance))

            # Update agent's model of the environment
            agent.update(recommendation-1, instance[-1])
            
    print("Successfully trained ", agent._name, '.', sep = '')
            
    return history[1:].astype(int)


def score(history, rolling = 50):
    """
    Plots cumulative reward, mean reward and rolling mean reward for a given training history.
    
    Inputs:
    - history : (h x 3)-dimensional array of samples seen during training.
    - rolling : number of steps to average across when computing moving average.
    
    Outputs: none.
    """
    n_arms = int(max(history[:, -2]))
    
    timesteps         = np.asarray([i+1 for i in range(history.shape[0])])
    cumulative_reward = np.cumsum(history[:, -1])
    mean_reward       = cumulative_reward / timesteps
    rolling_mean      = [np.mean(history[i-rolling:i, -1]) for i in range(rolling, history.shape[0])]
    counts            = np.asarray([history[history[:, -2] == i+1, :].shape[0] for i in range(n_arms)]) / history.shape[0]
    cumulative_regret = np.cumsum(1 - history[:, -1])
    mean_regret       = (timesteps - cumulative_reward) / timesteps
    max_gap           = (counts[np.where(counts == max(counts))] - counts[np.where(counts == min(counts))])[0]
        
    breakdown = []
    for j in range(n_arms):
        b = np.cumsum(history[:, -2] == j+1)
        breakdown.append(b / timesteps)
        
    breakdown_rolling = []
    for b in breakdown:
        breakdown_rolling.append([np.mean(b[i-rolling:i]) for i in range(rolling, len(b))])
    
    figure, _ = plt.subplots(6, 1)
    figure.tight_layout(pad = 4.0)
    
    plt.rcParams['figure.figsize'] = [15, 16]
    
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
    # print('Number of items liked by user     :', sum(history[:, -1]),)
    print('Total reward                      : ', cumulative_reward[-1], '/', history.shape[0], sep = '')

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