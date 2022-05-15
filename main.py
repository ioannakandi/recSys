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
import json
from flask import *
from flask_cors import CORS, cross_origin

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# App config.
app = Flask(__name__, static_url_path='',
            static_folder='templates',
            template_folder='templates')
DEBUG = True
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
CORS(app)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

ratings = pd.read_csv("DATA_S2/q1-dataset/user_shows_new1.txt")
#ratings.head()
#ratings.insert(0, 'userId', range(1, 1 + len(ratings)))
#ratings.to_csv('user_shows_new.txt', index=False)

#transform the given dataset
shows = pd.read_csv("DATA_S2/q1-dataset/shows_new1.txt")
#shows.head()
#conversion of dataset columns to rows
new_ratings=pd.melt(ratings,id_vars=["userId"], 
        var_name="show", 
        value_name="Watched")



#userBasedApproach = []


#new_ratings.to_csv('user_shows_new1.txt', index=False)

n_ratings = len(ratings)
#n_shows = len(shows)
n_shows = len(ratings['Show'].unique())
n_users = len(ratings['userId'].unique())

#print(f"Number of ratings: {n_ratings}")
#print(f"Number of unique Show's: {n_shows}")
#print(f"Number of unique users: {n_users}")
#print(f"Average shows watched per user: {round(n_ratings/n_users, 2)}")
#print(f"Average watched status per show: {round(n_ratings/n_shows, 2)}")


# =============================================================================
# user_freq = ratings[['userId', 'Show']].groupby('userId').count().reset_index()
# user_freq.columns = ['userId', 'n_ratings']
# user_freq.head()
# =============================================================================


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


"""
Find similar users using KNN
"""
def find_similar_users(userId, X, k, metric='cosine', show_distance=False):
	
	neighbour_ids = []
	
	user_ind = user_mapper[userId]
	user_vec = X[user_ind]
	k+=1
	kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
	kNN.fit(X)
	user_vec = user_vec.reshape(1,-1)
	neighbour = kNN.kneighbors(user_vec, return_distance=show_distance)
	for i in range(0,k):
		n = neighbour.item(i)
		neighbour_ids.append(user_inv_mapper[n])
	neighbour_ids.pop(0)
	return neighbour_ids


shows_titles = shows['Show']

show_title = 'The Situation Room with Wolf Blitzer'
userId = 1

#these are for testing
'''
similar_shows = find_similar_shows(show_title, X, k=10)
similar_users = find_similar_users(userId, X, k=3)

print(f"Since you watched {show_title}")
print(similar_shows)
print(similar_users)
watched_shows = ratings.loc[ratings['Watched'] == 1]
watched_shows = watched_shows.drop_duplicates('Show', keep='first')
for userId in similar_users:
    userBasedApproach.append(watched_shows.iloc[userId]['Show']) 
#convert list to JSON
userBasedApproachJson = json.dumps(userBasedApproach)
print(userBasedApproachJson)
'''

#this is the endpoint retrieving similar shows based on watched shows
@app.route("/getShows",methods=['GET', 'POST'])
@cross_origin()
def getShows():
    if request.method == 'GET':
        search_parameter = request.args.get('search_parameter')
        search_method = request.args.get('search_method')
        #check the search parameter to perform an item-based or user-based search
        if search_method == "item": 
            similar_shows = find_similar_shows(search_parameter, X, k=10)
            similar_shows_JSON = json.dumps(similar_shows)
            return Response(similar_shows_JSON, status=200, mimetype="application/json")
        elif search_method == "user": 
            #this is a list to store all the found shows
            userBasedApproach = []
            similar_users = find_similar_users(int(search_parameter), X, k=10)
            watched_shows = ratings.loc[ratings['Watched'] == 1]
            watched_shows = watched_shows.drop_duplicates('Show', keep='first')
            for userId in similar_users:
                userBasedApproach.append(watched_shows.iloc[userId]['Show']) 
            #convert list to JSON
            similar_shows_JSON = json.dumps(userBasedApproach)
            return Response(similar_shows_JSON, status=200, mimetype="application/json")
    return Response('{"status":"error"}', status=500, mimetype="application/json")



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)