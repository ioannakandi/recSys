# -*- coding: utf-8 -*-
"""
Authors: Ioanna Kandi & Konstantinos Mavrogiorgos
"""
# code
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = pd.read_csv("C:/Users/Komav/Desktop/recSys/DATA_S2/q1-dataset/user_shows_new1.txt")
#ratings.head()
#ratings.insert(0, 'userId', range(1, 1 + len(ratings)))
#ratings.to_csv('user_shows_new.txt', index=False)

#transform the given dataset
shows = pd.read_csv("C:/Users/Komav/Desktop/recSys/DATA_S2/q1-dataset/shows_new1.txt")
#shows.head()
#conversion of dataset columns to rows
new_ratings=pd.melt(ratings,id_vars=["userId"], 
        var_name="show", 
        value_name="Watched")


#new_ratings.to_csv('user_shows_new1.txt', index=False)

n_ratings = len(ratings)
#n_shows = len(shows)
n_shows = len(ratings['Show'].unique())
n_users = len(ratings['userId'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique Show's: {n_shows}")
print(f"Number of unique users: {n_users}")
#print(f"Average shows watched per user: {round(n_ratings/n_users, 2)}")
#print(f"Average watched status per show: {round(n_ratings/n_shows, 2)}")


user_freq = ratings[['userId', 'Show']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
user_freq.head()


# =============================================================================
# # Find Lowest and Highest rated shows:
# mean_rating = ratings.groupby('Show')[['rating']].mean()
# # Lowest rated shows
# lowest_rated = mean_rating['rating'].idxmin()
# shows.loc[shows['Show'] == lowest_rated]
# # Highest rated shows
# highest_rated = mean_rating['rating'].idxmax()
# shows.loc[shows['Show'] == highest_rated]
# # show number of people who rated shows rated show highest
# ratings[ratings['Show']==highest_rated]
# # show number of people who rated shows rated show lowest
# ratings[ratings['Show']==lowest_rated]
# 
# ## the above shows has very low dataset. We will use bayesian average
# show_stats = ratings.groupby('Show')[['rating']].agg(['count', 'mean'])
# show_stats.columns = show_stats.columns.droplevel()
# =============================================================================

# Now, we create user-item matrix using scipy csr matrix
from scipy.sparse import csr_matrix

def create_matrix(df):
	
	N = len(df['userId'].unique())
	M = len(df['Show'].unique())
	
	# Map Ids to indices
	user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
	show_mapper = dict(zip(np.unique(df["Show"]), list(range(M))))
	
	# Map indices to IDs
	user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
	show_inv_mapper = dict(zip(list(range(M)), np.unique(df["Show"])))
	
	user_index = [user_mapper[i] for i in df['userId']]
	show_index = [show_mapper[i] for i in df['Show']]

	X = csr_matrix((df["Watched"], (show_index, user_index)), shape=(M, N))
	
	return X, user_mapper, show_mapper, user_inv_mapper, show_inv_mapper

X, user_mapper, show_mapper, user_inv_mapper, show_inv_mapper = create_matrix(ratings)

from sklearn.neighbors import NearestNeighbors
"""
Find similar shows using KNN
"""
def find_similar_shows(show_title, X, k, metric='cosine', show_distance=False):
	
	neighbour_ids = []
	
	show_ind = show_mapper[show_title]
	show_vec = X[show_ind]
	k+=1
	kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
	kNN.fit(X)
	show_vec = show_vec.reshape(1,-1)
	neighbour = kNN.kneighbors(show_vec, return_distance=show_distance)
	for i in range(0,k):
		n = neighbour.item(i)
		neighbour_ids.append(show_inv_mapper[n])
	neighbour_ids.pop(0)
	return neighbour_ids


shows_titles = shows['Show']

show_title = 'The Situation Room with Wolf Blitzer'

similar_shows = find_similar_shows(show_title, X, k=10)

print(f"Since you watched {show_title}")
print(similar_shows)

