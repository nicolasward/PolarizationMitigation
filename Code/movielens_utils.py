import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def getdata(binary_ratings = False):
    """
    --- Original genres: ---
    
    0  : Unknown
    1  : Action
    2  : Adventure
    3  : Animation
    4  : Children's
    5  : Comedy
    6  : Crime
    7  : Documentary
    8  : Drama
    9  : Fantasy
    10 : Film-Noir
    11 : Horror
    12 : Musical
    13 : Mystery
    14 : Romance
    15 : Sci-Fi
    16 : Thriller
    17 : War
    18 : Western
    
    
    --- Meta-genres: ---
    
    1 : Action / Adventure / War
    2 : Children / Animation
    3 : Comedy / Musical 
    4 : Thriller / Film-noir / Crime / Western / Mystery
    5 : Romance / Drama
    6 : Fantasy / Sci-Fi
    7 : Horror
    
    Removed genres: 'unknown' and 'documentary'.
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
    
    
    # ----- Group movies by meta-genre: [movie_id, meta genre] ----- #
    
    meta_genres = { 1: [2, 3, 18],
                    2: [4, 5],
                    3: [6, 13],
                    4: [7, 11, 17, 19, 14],
                    5: [8],
                    6: [9, 15],
                    7: [10, 16],
                    8: [12] }
    
    meta_movie_genres = np.zeros((1, 2))
    
    for g in meta_genres:
        data = movie_genres[:, meta_genres[g]]
        pos  = list(set(np.where(data == 1)[0]))
        temp = np.hstack((movie_genres[pos, 0].reshape((1, -1)).T, np.asarray([g] * len(pos)).reshape((1, -1)).T))
        meta_movie_genres = np.vstack((meta_movie_genres, temp))
            
    
    # ------ Merge Users, Movies, Ratings and Meta Genres: [user_id, movie_id, rating, meta genre] ------ #

    movielens = np.zeros((1, 4))
    for movie in meta_movie_genres:
        movie_id, movie_genre = movie
        data = movielens_raw[movielens_raw[:, 1] == movie_id, :]
        n    = data.shape[0]
        cat  = np.asarray([movie_genre] * n).reshape((1, -1)).T
        movielens = np.vstack((movielens, np.hstack((data, cat))))
    movielens = movielens[1:,]
        
    
    # ----- Delete movies belonging to 'unknown' and 'documentary' genres ----- #

    unknown_genre_ids = movie_genres[movie_genres[:, 1] > 0][:, 0]
    ids               = np.where(np.in1d(movielens[:, 1], unknown_genre_ids) == True)[0]
    movielens         = np.delete(movielens, np.s_[ids], axis = 0)
    
    documentary_ids   = movie_genres[movie_genres[:, 8] > 0][:, 0]
    ids               = np.where(np.in1d(movielens[:, 1], documentary_ids) == True)[0]
    movielens         = np.delete(movielens, np.s_[ids], axis = 0)
    
    movielens[movielens[:, -1] > 5, -1] -= 1
    
    if binary_ratings:
        movielens[:, 2] = [1 if element >= 4 else 0 for element in movielens[:, 2]]
 
        
    # Swap the 2 last columns to obtain final formatting: [user_id, movie_id, meta genre, rating]
        
    movielens[:, [2, 3]] = movielens[:, [3, 2]]
        
    return movielens.astype(int)


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
        print('Total number of items in dataset :', n)
        
    figure, _ = plt.subplots(2, 1)
    figure.tight_layout(pad = 4.0)
   
    plt.subplot(2, 1, 1)
    plt.bar(list(genres), means)
    plt.xticks(list(range(1, len(list(genres))+1)))
    if user_id < 0:
        plt.title('Proportion of each movie genre in the full dataset')
    else:
        plt.title('Proportion of each movie genre rated by user %i' % user_id)
    plt.xlabel('Genre')
    plt.ylabel('Proportion')
    
    print_ratios = ['%.2f' % elem for elem in like_ratios]
    print_means  = ['%.2f' % elem for elem in means]
    
    print('Meta genres proportion of ratings :', [float(i) for i in print_means])
    print('Meta genres proportion of likes   :', [float(i) for i in print_ratios])

    plt.subplot(2, 1, 2)
    plt.bar(list(genres), like_ratios)
    plt.xticks(list(range(1, len(list(genres))+1)))
    if user_id < 0:
        plt.title('Per-genre like ratio for all users combined.')
    else:
        plt.title('Per-genre like ratio for user %i' % user_id)
    plt.xlabel('Genre')
    plt.ylabel('Like ratio')
    
    plt.show()
    
    return user


def dataset_stats(user):
    
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
        
    print('Total number of items in dataset :', n)
        
    figure, _ = plt.subplots(2, 1)
    figure.tight_layout(pad = 4.0)
   
    plt.subplot(2, 1, 1)
    plt.bar(list(genres), means)
    plt.xticks(list(range(1, len(list(genres))+1)))
    plt.title('Proportion of each movie genre in the full dataset')
    plt.xlabel('Genre')
    plt.ylabel('Proportion')
    
    print_ratios = ['%.2f' % elem for elem in like_ratios]
    print_means  = ['%.2f' % elem for elem in means]
    
    print('Meta genres proportion of ratings :', [float(i) for i in print_means])
    print('Meta genres proportion of likes   :', [float(i) for i in print_ratios])

    plt.subplot(2, 1, 2)
    plt.bar(list(genres), like_ratios)
    plt.xticks(list(range(1, len(list(genres))+1)))
    plt.title('Per-genre like ratio for full dataset')
    plt.xlabel('Genre')
    plt.ylabel('Like ratio')
    
    plt.show()
    
    
def user_counts(data):
    
    user_counts = dict.fromkeys(set(data[:, 0]), 0)
    for i in data[:, 0]:
        user_counts[i] += 1
        
    return {k: v for k, v in sorted(user_counts.items(), key = lambda item: item[1], reverse = True)}

    
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