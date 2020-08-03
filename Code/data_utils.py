from matplotlib import pyplot as plt
import numpy as np

def new_user(user_id, n = 1000, n_arms = 5, preferences = [0.9, 0.5, 0.5, 0.5, 0.5], print_stats = False):
    """
    Generates data for a new user given their preferences.
    
    Inputs:
    - user_id     : integer to identify the user.
    - n           : number of samples (rows) in the data to generate.
    - n_arms      : number of arms (categories/degrees of polarisation of content).
    - preferences : numbers in [0,1]; number i corresponds to the probability of the user liking a random item from group i.
    
    Outputs:
    - data : (n x 3)-dimensional array containing preferences generated for the user.
    """
    if len(preferences) != n_arms:
        print("!!! Preferences vector doesn't match number of arms !!!")
    
    data       = np.zeros((n, 3))    
    data[:, 0] = user_id
    data[:, 1] = np.random.choice(range(1, n_arms+1), size = n)
    data[:, 2] = [np.random.choice(range(2), size = 1, p = [1 - preferences[i-1], preferences[i-1]]) for i in data[:, 1].astype(int)]
    
    if print_stats:
        prop = np.zeros((n_arms, 2))
        prop[:, 0] = range(1, n_arms + 1)
        prop[:, 1] = [sum(data[:, 1] == j+1) / n for j in range(n_arms)]
        
        group_likes = np.zeros((n_arms, 2))
        group_likes[:, 0] = range(1, n_arms + 1)
        group_likes[:, 1] = [sum(data[data[:, 1] == j+1][:, 2]) / data[data[:, 1] == j+1].shape[0] for j in range(n_arms)]

        overall_likes = np.zeros((n_arms, 2))
        overall_likes[:, 0] = range(1, n_arms + 1)
        overall_likes[:, 1] = [sum(data[data[:, 1] == j+1][:, 2]) / sum(data[:, 2]) for j in range(n_arms)]        
        
        print("--- Stats for user", user_id, ": --- \n")
        print("Group representation: \n", prop)
        print("\n --------------------- \n")
        print("Proportion of liked items inside each group: \n", group_likes)
        print("\n --------------------- \n")
        print("Proportion of all likes allocated to each group: \n", overall_likes)
        print("\n --------------------- \n")
        print("User data: \n", data.astype(int))
        
    return data.astype(int)

def merge_users(user1, user2, shuffle = False):
    """
    Merges data from 2 users.
    
    Inputs:
    - user1   : (n x 3)-dimensional array containing data for first user.
    - user2   : (m x 3)-dimensional array containing data for second user.
    - shuffle : boolean (default: False) which determines whether to shuffle the rows of the merged dataset or not.
    
    Outputs:
    - ((n + m) x 3)-dimensional array containing merged data of both users.
    """
    result = np.concatenate((user1, user2))
    if shuffle: np.random.shuffle(result)
        
    return result