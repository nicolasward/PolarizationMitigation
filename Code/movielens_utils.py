import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def getdata(binary_ratings = False):
    """
    Clustered movie genres:
    
    1 : Action / Adventure / War
    2 : Children / Animation
    3 : Comedy / Musical 
    4 : Thriller / Film-noir / Crime / Western
    5 : Documentary
    6 : Romance / Drama
    7 : Fantasy / Sci-Fi
    8 : Horror
    9 : Mystery
    
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
    
    if binary_ratings:
        movielens_final[:, 2] = [1 if element >= 4 else 0 for element in movielens_final[:, 2]]
        
        
    # Swap the 2 last columns to obtain final formatting: [user_id, movie_id, genre, rating]
        
    movielens_final[:, [2, 3]] = movielens_final[:, [3, 2]]
    
    movielens_final[:, -2] += 1
        
    return movielens_final.astype(int)


def get_user(dataset, user_id = -1):
        
    # Isolate user data
    if user_id > 0:
        user = dataset[dataset[:, 0] == user_id, :]
    else:
        user = dataset
        
    # Initialize counts list and number of items for this user
    like_ratios = []
    means = []
    likes = []
    n = user.shape[0]
    genres = set(user[:, -2])
    
    # For each category, calculate and store the average number of items rated by the user
    for j in genres:
        genre_j = user[user[:, -2] == j, :]
        n_items = genre_j.shape[0]
        n_likes = sum(genre_j[:, -1])
        likes.append(n_likes)
        ratio   = n_likes / n_items if n_items != 0 else 0
        like_ratios.append(ratio)
        means.append(n_items / n)
        
    if user_id > 0:
        print('Total number of items rated by user %i :' % user_id, n)
        print('Total number of items liked by user %i :' % user_id, sum(likes))
    else:
        print('Total number of items in dataset :' % user_id, n)
        
    figure, _ = plt.subplots(2, 1)
    figure.tight_layout(pad = 4.0)
   
    plt.subplot(2, 1, 1)
    plt.bar(list(genres), means)
    plt.xticks(list(range(1, 10)))
    if user_id < 0:
        plt.title('Proportion of each movie genre in the full dataset')
    else:
        plt.title('Proportion of each movie genre rated by user %i' % user_id)
    plt.xlabel('Genres')
    plt.ylabel('Proportion')

    plt.subplot(2, 1, 2)
    plt.bar(list(genres), like_ratios)
    plt.xticks(list(range(1, 10)))
    if user_id < 0:
        plt.title('Per-genre like ratio for all users combined.')
    else:
        plt.title('Per-genre like ratio for user %i' % user_id)
    plt.xlabel('Genres')
    plt.ylabel('Like ratio')
    
    plt.show()
    
    if user_id > 0:
        return user
    else:
        return like_ratios
    
def like_gaps(dataset):
    
    genres = set(dataset[:, -2])
    user_ids = set(dataset[:, 0])
    gaps = {}
    
    for user_id in user_ids:
        user = dataset[dataset[:, 0] == user_id, :]
        like_ratios = []
        for j in genres:
            genre_j = user[user[:, -2] == j, :]
            n_items = genre_j.shape[0]
            n_likes = sum(genre_j[:, -1])
            ratio   = n_likes / n_items if n_items != 0 else 0
            like_ratios.append(ratio)
        s = sorted(like_ratios)[-2:]
        gaps[user_id] = s[-1] - s[-2]
    
    return {k: v for k, v in sorted(gaps.items(), key = lambda item: item[1], reverse = True)}