import numpy as np
from operator import itemgetter

def getdata():
    
    # ------ Get User Ratings: [user_id, movie_id, rating] ------ #

    f = open('../Data/u.data', 'r+')
    movielens_raw = np.array([0, 0, 0, 0])
    for l in f: 
        movielens_raw = np.vstack((movielens_raw, list(map(int, l.strip().split("\t")))))
    movielens_raw = movielens_raw[1:, :3]
       

    # ------ Get Movie Genres: [movie_id, 19 movie categories] ------ #

    f = open('../Data/u.item', 'r+', encoding = "ISO-8859-1")
    movie_genres = np.zeros((1, 20))
    indices   = list(range(5, 24))
    indices.insert(0, 0)
    for l in f:
        line = l.strip().split("|")
        line = list(itemgetter(*indices)(line))
        movie_genres = np.vstack((movie_genres, list(map(int, line)))) 
    movie_genres = movie_genres[1:].astype(int)
       
    
    # ----- Merge overlapping categories ----- #

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
    
    
    # ------ Merge Genres and Ratings to obtain final dataset: [user_id, movie_id, rating, 19 movie categories] ------ #

    movielens = np.zeros((1, 13))
    for i in movie_genres[:, 0]:
        n = movielens_raw[movielens_raw[:, 1] == i, :].shape[0]
        categories = np.expand_dims(movie_genres[i-1, 1:], axis = 0)
        categories = np.repeat(categories, repeats = n, axis = 0)
        movielens = np.vstack((movielens, np.hstack((movielens_raw[movielens_raw[:, 1] == i, :], categories))))
    movielens = movielens[1:,]
    
    
    # ----- Delete movies belonging to unknown genre as well as unknown genre column ----- #

    unknown_genre_ids = movie_genres[movie_genres[:, 1] > 0][:, 0]
    
    ids = np.where(np.in1d(movielens[:, 1], unknown_genre_ids) == True)[0]
    
    movielens = np.delete(movielens, np.s_[ids], axis = 0)
    movielens = np.delete(movielens,   np.s_[3], axis = 1)
        
    return movielens.astype(int)