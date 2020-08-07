import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def getdata():
    """
    Clustered movie genres:
    
    0 : Action / Adventure / War
    1 : Children / Animation
    2 : Comedy / Musical 
    3 : Thriller / Film-noir / Crime / Western
    4 : Documentary
    5 : Romance / Drama
    6 : Fantasy / Sci-Fi
    7 : Horror
    8 : Mystery
    
    """
    
    # ------ Get User Ratings: [user_id, movie_id, rating] ------ #

    f = open('../Data/u.data', 'r+')
    movielens_raw = np.array([0, 0, 0, 0])
    for l in f: 
        movielens_raw = np.vstack((movielens_raw, list(map(int, l.strip().split("\t")))))
    movielens_raw = movielens_raw[1:, :3]
       

    # ------ Get Movie Genres: [movie_id, 19 movie genres] ------ #

    f = open('../Data/u.item', 'r+', encoding = "ISO-8859-1")
    movie_genres = np.zeros((1, 20))
    indices   = list(range(5, 24))
    indices.insert(0, 0)
    for l in f:
        line = l.strip().split("|")
        line = list(itemgetter(*indices)(line))
        movie_genres = np.vstack((movie_genres, list(map(int, line)))) 
    movie_genres = movie_genres[1:].astype(int)
       
    
    # ----- Merge overlapping movie genres ----- #

    movie_genres[:, 2] += movie_genres[:, 3] + movie_genres[:, 18]
    movie_genres[movie_genres[:, 2] > 1, 2] = 1

    movie_genres[:, 7] += movie_genres[:, 11] + movie_genres[:, 17] + movie_genres[:, 19]
    movie_genres[movie_genres[:, 7] > 1, 7] = 1

    movie_genres[:, 9] += movie_genres[:, 15]
    movie_genres[movie_genres[:, 8] > 1, 8] = 1

    movie_genres[:, 10] += movie_genres[:, 16]
    movie_genres[movie_genres[:, 10] > 1, 10] = 1

    movie_genres[:, 4] += movie_genres[:, 5]
    movie_genres[movie_genres[:, 4] > 1, 4] = 1

    movie_genres[:, 6] += movie_genres[:, 13]
    movie_genres[movie_genres[:, 6] > 1, 6] = 1

    movie_genres = np.delete(movie_genres, np.s_[3, 5, 11, 13, 15, 16, 17, 18, 19], axis = 1)
    
    
    # ------ Store new clustered movie genres in a dictionary ------ #
    
    genres = {}
    for line in movie_genres:
        genres[line[0]] = np.where((line[2:] > 0))[0]
        
    
    # ------ Merge Genres and Ratings: [user_id, movie_id, rating, 9 movie genres] ------ #

    movielens = np.zeros((1, 13))
    for i in movie_genres[:, 0]:
        n = movielens_raw[movielens_raw[:, 1] == i, :].shape[0]
        categories = np.expand_dims(movie_genres[i-1, 1:], axis = 0)
        categories = np.repeat(categories, repeats = n, axis = 0)
        movielens = np.vstack((movielens, np.hstack((movielens_raw[movielens_raw[:, 1] == i, :], categories))))
    movielens = movielens[1:,]
    
    
    # ------ Summarize movie genre information with a single number: [user_id, movie_id, rating, genre] ------ #
    
    movielens_final = np.zeros((1, 4))
    for i in genres.keys():
        buffer = movielens[movielens[:, 1] == i]
        n      = buffer.shape[0]
        for j in genres[i]:
            category = np.asarray([j] * n)
            category = np.expand_dims(category, axis = 1)
            movielens_final = np.vstack((movielens_final, np.hstack((buffer[:, :3], category))))
    movielens_final = movielens_final[1:,]
    
    
    # ----- Delete movies belonging to 'unknown' genre as well as 'unknown' genre column ----- #

    unknown_genre_ids = movie_genres[movie_genres[:, 1] > 0][:, 0]
    ids               = np.where(np.in1d(movielens_final[:, 1], unknown_genre_ids) == True)[0]
    movielens_final   = np.delete(movielens_final, np.s_[ids], axis = 0)
        
    return movielens_final.astype(int)


def user_stats(dataset, user_id = -1, plot = False):
    
    # Isolate user data
    if user_id >= 0:
        user = dataset[dataset[:, 0] == user_id, :]
    else:
        user = dataset
        
    # Initialize counts list and number of items for this user
    counts = []
    means  = []
    n = user.shape[0]
    genres = set(user[:, 3])
    
    # For each category, calculate and store the average number of items rated by the user
    for j in genres:
        k = sum(user[:, 3] == j)
        counts.append(k)
        means.append(k/n)
        
    if plot:
        plt.rcParams['figure.figsize'] = [15, 6]
        plt.bar(list(genres), means)
        plt.xticks(list(range(9)))
        if user_id < 0:
            plt.title('Proportion of each movie genre in the entire dataset')
        else:
            plt.title('Proportion of each movie genre rated by user %i' % user_id)
        plt.xlabel('Genres')
        plt.ylabel('Proportion')
        
    return counts, means