#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd


# In[67]:


df= pd.read_csv('Mall_Customers.csv')


# In[68]:


#here ther is no label provided
#work on annual Income (k$) and Spending Score (1-100)
df.head()


# In[69]:


df.tail()


# In[70]:


df.shape


# In[71]:


df.describe()


# In[72]:


df.info()


# In[73]:


#check for null value
df.isnull().sum()


# In[74]:


import matplotlib.pyplot as plt


# In[75]:


#Do EDA
#from data points we can say there will be 5 clusters
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])


# In[76]:


X=df[['Annual Income (k$)','Spending Score (1-100)','Age']].values


# In[77]:


from sklearn.cluster import KMeans


# In[78]:


#SELECT CLUSTER SIZE AS 5; Default is 8; check with signature Shift + tab 
model=KMeans(n_clusters=5,random_state=0)


# In[79]:


model.fit(X)


# In[80]:


y=model.predict(X)


# In[81]:


#check clusters as output; It is showing cluster numbers starts from 0 ie 0 to 4 
y


# In[82]:


#check number of values or data points in each cluster
#output shows at 0th cluster there are 35 data points, ata 1st cluster 81 and so on
import numpy as np
np.unique(y,return_counts=True)


# In[83]:


np.sum([35, 81, 39, 22, 23])
#df.shape


# In[84]:


df.shape


# In[85]:


#check for first cluster how many from 0th columns ie Annual Income (k$)
X[y==0,0]


# In[86]:


#check how many from 1st columns ie Spending Score (1-100)
X[y==0,1]


# In[87]:


#check for 4th cluster how many from 0th columns ie Annual Income (k$)
X[y==3,0]


# In[88]:


#check for 4th cluster how many from 0th columns ie Spending Score (1-100)
X[y==3,1]


# In[89]:


#show data points of 1st cluster with Annual Income (k$) and Spending Score (1-100)
plt.scatter(X[y==0,0],X[y==0,1])


# In[90]:


#show data points of 2nd cluster with Annual Income (k$) and Spending Score (1-100)
plt.scatter(X[y==1,0],X[y==1,1])


# In[91]:


plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])


# In[92]:


#show clusters combine
for i in range(5):
    plt.scatter(X[y==i,0],X[y==i,1])


# In[93]:


#assignment change the value of clusters to 6 ie model=KMeans(n_clusters=6,random_state=0)


# In[94]:


#show cluster centers or centroid for Annual Income (k$) and Spending Score (1-100)
model.cluster_centers_


# In[95]:


#show clusters combine; s is used for size; s=150
for i in range(5):
    plt.scatter(X[y==i,0],X[y==i,1])
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=150,color='yellow')


# In[96]:


#sklearn.metrics.silhouette_score as metric
#- silhouette score is btwn -1 to 1
#if silhouette score is near to the -1 mean sprawling(spreading out over a language area in an untidy or irregular way), overlapped clusters and 
#if silhouette score is near 1 means Tight, well-separated clusters

from sklearn.metrics import silhouette_score
silhouette_score(X,y)


# In[97]:


#E-low method
#sum of squared errors SSE is calculated for every number and it is stored at parameter called inertia
#inertia = sum if squared distances of sample to their closest cluster center, weighted by the sample weights if provided.
#-if we are getting less number for intertia then we will select that number of cluster
model.inertia_


# In[98]:


#how many clusters we can make
len(df)
np.sqrt(200)


# In[99]:


#Elbow nethod
k=range(2,15)
sse=[]
for i in k:
    demo_model=KMeans(n_clusters=i,random_state=0).fit(X)
    sse.append(demo_model.inertia_)
plt.scatter(k,sse)


# In[103]:


#graph with connecting points
k=range(2,15)
sse=[]
for i in k:
    demo_model=KMeans(n_clusters=i,random_state=0).fit(X)
    sse.append(demo_model.inertia_)
plt.scatter(k,sse)
plt.plot(k,sse)


# In[101]:


# in above output graph at 5 location graph is decreasing so elbow or bend  is at 5 
# so we will select 5 number of clusers as optimum clusters


# In[102]:


#show silhouette_score with cluster # from output we found that 5 cluster, Score 0.553931997444648 is the maximum score
#so we can finalize the cluster size=5
k=range(2,15)
sse=[]
for i in k:
    demo_model=KMeans(n_clusters=i,random_state=0).fit(X)
    sse.append(demo_model.inertia_)
    y=demo_model.predict(X)
    print(f"{i}Cluster,Score{silhouette_score(X,y)}")
    plt.bar(i,silhouette_score(X,y))
plt.scatter(k,sse)
plt.plot(k,sse)
plt.show()


# In[ ]:




