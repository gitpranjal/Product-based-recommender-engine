
# coding: utf-8

# # Task2:  Amazon prime movie recommendations to users
# Big ecommerce firms like Amazon, use product based recommendation system. This means that, once, a user has purchased a product, recommendations to him are made based upon the produts adopted and reviwed by the other users. In this program, the dataset containing movies and their reviews, ratings by the users on amazon Prime are used. Once a user watches a movie, he is recommended as set of movies. For this purpose, the correlation of the movie's ratings columns by the users,with other movies' ratings is calculated. And then , the other movies are sorted based upon the correaltions. The most correlatied movies are then filtered out. 

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


column_names=["user_id", "item_id", "rating", "timestamp"]
df=pd.read_csv("u.data", sep="\t", names=column_names)
df=pd.read_csv("u.data", sep="\t", names=column_names)


# In[84]:


df.head()
#User_id is the id of the person who watched the movie( referred by item_id) at time=timestamp and rated it to value in rating


# In[85]:


movie_titles=pd.read_csv("Movie_Id_Titles")


# In[86]:


movie_titles.head()


# In[87]:


df=pd.merge(df,movie_titles, on="item_id")


# In[88]:


df.head()


# # Average Rating of movies

# In[89]:


df.groupby("title").mean()["rating"].sort_values(ascending=False)


# # Most watched movies

# In[90]:


df.groupby("title").count()


# In[92]:


ratings=pd.DataFrame(df.groupby("title")["rating"].mean())


# In[93]:


ratings.head()


# Rating carries value, only if there are lots of people giving that average rating

# In[94]:


ratings["num of ratings"]=df.groupby("title")["rating"].count()


# In[95]:


ratings.head()


# In[96]:


ratings["num of ratings"].hist(bins=70)
#Shows most of the movies don't have a huge viewer ship and hence rating. Most people don't rate the movies


# In[97]:


ratings["rating"].hist(bins=70)
#Shows, most of the movies are rated between 2.5 to 3.5


# In[98]:


sns.jointplot(x="rating", y="num of ratings", data=ratings)


# In[99]:


sns.lmplot(x="rating", y="num of ratings", data=ratings)
#Shows that more the rating of the movie, better will be the viewership of it and hence, more people will review it


# # Building Recommender System

# In[130]:


movie_matrix=df.pivot_table(index="user_id", columns="title", values="rating")


# In[131]:


movie_matrix.head()
#Lots of missing values because most people havn't watched most of the movies


# In[132]:


ratings=ratings.sort_values("num of ratings", ascending=False).head(10)


# In[141]:


ratings


# # Enter the movie , a user watched recently
# Based upon this movie, he will be made recommendations

# In[160]:


movie=input()


# In[162]:


selected_movie_ratings=movie_matrix[str(movie)]


# In[163]:


selected_movie_reccomendations=pd.DataFrame(movie_matrix.corrwith(selected_movie_ratings), columns=["correlations"])


# In[165]:


selected_movie_reccomendations.sort_values(by="correlations", ascending=False).head(10)

