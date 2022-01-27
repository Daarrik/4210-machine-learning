#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: clustering.py
# SPECIFICATION: Finds the best number of clusters k that
# maximizes Silhouette coefficient. Uses the same k to 
# calculate Homogeneity score of the clusters and the
# Homogeneity score of Agglomerative clustering.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
silhoutte_coef_list = []
for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)
     silhouette_coef = silhouette_score(X_training, kmeans.labels_)
     silhoutte_coef_list.append(silhouette_coef)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
ks = [k for k in range(2, 21)]
plt.plot(ks, silhoutte_coef_list)
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
testing = pd.read_csv('testing_data.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(testing.values).reshape(1, -1)[0]
#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print(f'K-Means Homogeneity Score = {metrics.homogeneity_score(labels, kmeans.labels_)}')
#rung agglomerative clustering now by using the best value o k calculated before by kmeans
max_silhouette = (max(silhoutte_coef_list))
best_k = ks[silhoutte_coef_list.index(max_silhouette)]

#Do it:
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
# print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
print(f'Agglomerative Clustering Homogeneity Score = {metrics.homogeneity_score(labels, agg.labels_)}')
