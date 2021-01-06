from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plotly.figure_factory as ff
from scipy.cluster import hierarchy

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import davies_bouldin_score, silhouette_score

iris = load_iris()

# Define data and labels
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

print(x)
print(y)


# Data visualization
colors = np.array(['red', 'green', 'blue'])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Petal Length'], x['Petal Width'], c=colors[iris.target], s=50)
axes[0].set_xlabel('Sepal length', fontsize=14)
axes[0].set_ylabel('Sepal width', fontsize=14)
axes[1].set_xlabel('Petal length', fontsize=14)
axes[1].set_ylabel('Petal width', fontsize=14)

plt.show()

# -----------------Elbow Method for Iris------------------
# step 1: pre-clustering
# k-means clustering
cost = []
for i in range(1, 7):
    km = KMeans(n_clusters=i, max_iter=500)
    # perform k-means clustering on data X
    km.fit(x)
    # calculates squared error for the clustered points
    cost.append(km.inertia_)

# plot the cost against k values
plt.plot(range(1, 7), cost, color='b', linewidth='4')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.title("Elbow method iris")
plt.show()

# step2： cluster analysis
# -----------------Dimensionality reduction to 2 dimensions using PCA-----------------
pca = PCA(n_components=2)
X_iris_pca = pca.fit(x).transform(x)
X_iris_pca = pd.DataFrame(X_iris_pca, columns=['PC-1', 'PC-2'])
print(X_iris_pca)

# -----------------Dimensionality reduction to 2 dimensions using LDA-----------------
lda = LinearDiscriminantAnalysis(n_components=2)
X_iris_lda = lda.fit(x, y).transform(x)
X_iris_lda = pd.DataFrame(X_iris_lda, columns=['PC-1', 'PC-2'])
print(X_iris_lda)

# ------------------ KMeans for pca iris-------------------------
km = KMeans(n_clusters=2, n_jobs=3, random_state=111)
km.fit(X_iris_pca)
centroids = pd.DataFrame(km.cluster_centers_, columns=['PC-1', 'PC-2'])
print(centroids)

# prediction label colors
color1 = np.array(['green', 'red', 'blue'])
pred_y = pd.DataFrame(km.labels_, columns=['Target'])

# data visualization pca - before and after clustering for Sepal length vs Sepal width
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(X_iris_pca['PC-1'], X_iris_pca['PC-2'], c=colors[y['Target']], s=50)
axes[1].scatter(X_iris_pca['PC-1'], X_iris_pca['PC-2'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['PC-1'], centroids['PC-2'], c='k', s=70)
axes[0].set_xlabel('PC-1', fontsize=14)
axes[0].set_ylabel('PC-2', fontsize=14)
axes[0].set_title('Before K-Means clustering iris PCA')
axes[1].set_xlabel('PC-1', fontsize=14)
axes[1].set_xlabel('PC-2', fontsize=14)
axes[1].set_title('After K-Means clustering iris PCA')
plt.show()

# --------Hierarchical clustering dissimilarity euclidean pca------------
linkage = hierarchy.linkage(X_iris_pca, metric='euclidean')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Dissimilarity PCA")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)

# --------Hierarchical clustering similarity cosine pca------------
linkage = hierarchy.linkage(X_iris_pca, metric='cosine')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Similarity PCA")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)
plt.show()

# ------------------ KMeans for lda iris-------------------------
km = KMeans(n_clusters=2, n_jobs=2, random_state=111)
km.fit(X_iris_lda)
centroids = pd.DataFrame(km.cluster_centers_, columns=['PC-1', 'PC-2'])
print(centroids)
# prediction label colors
color1 = np.array(['green', 'red', 'blue'])
pred_y = pd.DataFrame(km.labels_, columns=['Target'])

# data visualization pca - before and after clustering for Sepal length vs Sepal width
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(X_iris_lda['PC-1'], X_iris_lda['PC-2'], c=colors[y['Target']], s=50)
axes[1].scatter(X_iris_lda['PC-1'], X_iris_lda['PC-2'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['PC-1'], centroids['PC-2'], c='k', s=70)
axes[0].set_xlabel('PC-1', fontsize=14)
axes[0].set_ylabel('PC-2', fontsize=14)
axes[0].set_title('Before K-Means clustering iris LDA')
axes[1].set_xlabel('PC-1', fontsize=14)
axes[1].set_xlabel('PC-2', fontsize=14)
axes[1].set_title('After K-Means clustering iris LDA')
plt.show()

# --------Hierarchical clustering dissimilarity euclidean lda------------
linkage = hierarchy.linkage(X_iris_lda, metric='euclidean')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Dissimilarity LDA")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)

# --------Hierarchical clustering similarity cosine lda------------
linkage = hierarchy.linkage(X_iris_lda, metric='cosine')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Similarity LDA")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)
plt.show()

# ----------------KMeans without dimensionality reduction-----------------
km = KMeans(n_clusters=2, n_jobs=2, random_state=1000)
km.fit(x)
centroids = pd.DataFrame(km.cluster_centers_, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
print(centroids)
# prediction label colors
color1 = np.array(['green', 'red', 'blue'])
pred_y = pd.DataFrame(km.labels_, columns=['Target'])

# data visualization pca - before and after clustering for Sepal length vs Sepal width
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Sepal Length'], x['Sepal Width'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['Sepal Length'], centroids['Sepal Width'], c='k', s=70)
axes[0].set_xlabel('Sepal length', fontsize=14)
axes[0].set_ylabel('Sepal width', fontsize=14)
axes[0].set_title('Before K-Means clustering Sepal')
axes[1].set_xlabel('Sepal length', fontsize=14)
axes[1].set_xlabel('Sepal width', fontsize=14)
axes[1].set_title('After K-Means clustering Sepal')

# data visualization - before and after clustering for petal length vs petal width
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Petal Length'], x['Petal Width'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['Petal Length'], centroids['Petal Width'], c='k', s=70)
axes[0].set_xlabel('Petal length', fontsize=14)
axes[0].set_ylabel('Petal width', fontsize=14)
axes[0].set_title('Before K-Means clustering Petal')
axes[1].set_xlabel('Petal length', fontsize=14)
axes[1].set_xlabel('Petal width', fontsize=14)
axes[1].set_title('After K-Means clustering Petal')
plt.show()

# ----------------Hierarchical dissimilarity without dimensionality reduction-----------------
linkage = hierarchy.linkage(x, metric='euclidean')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Dissimilarity")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)

# ----------------Hierarchical similarity without dimensionality reduction-----------------
linkage = hierarchy.linkage(x, metric='cosine')
# plot the dendrogram
fig = plt.figure(figsize=(20, 5))
plt.title("Hierarchical Similarity")
s = hierarchy.dendrogram(linkage, leaf_font_size=12)
plt.show()

# ----------------Davies Bouldin and Silhouette index-------------------
# to generate a range of cluster validity scores
no_clusters = [2, 3, 4, 5, 6]
# cluster validity over a range of clusters
for i in no_clusters:
    # perform k-means
    km1 = KMeans(n_clusters=i)
    clabel = km1.fit_predict(x)
    # choose the minimum value among all of them for DB
    # ....... maximum value among all of them for Silhouette index

    # DB index
    print('Davis Bouldin index for cluster =', i, 'is', davies_bouldin_score(x, clabel))
    # The silhouette_score gives the average value for all the samples
    print('Silhouette index for clusters =', i, 'is', silhouette_score(x, clabel))


























#
# # k-means clustering
# km = KMeans(n_clusters=3, n_jobs=3, random_state=111)
# # perform k-means clustering on data X
# km.fit(x)
#
# # display centroids
# centroids = pd.DataFrame(km.cluster_centers_, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
# print(centroids)
#
# # prediction label colors
# color1 = np.array(['green', 'red', 'blue'])
# pred_y = pd.DataFrame(km.labels_, columns=['Target'])
#
# # # data visualization - before and after clustering for sepal length vs sepal width
# # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# # axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
# # axes[1].scatter(x['Sepal Length'], x['Sepal Width'], c=color1[pred_y['Target']], s=50)
# # axes[1].scatter(centroids['Sepal Length'], centroids['Sepal Width'], c='k', s=70)
# # axes[0].set_xlabel('Sepal length', fontsize=14)
# # axes[0].set_ylabel('Sepal width', fontsize=14)
# # axes[0].set_title('Before K-Means clustering')
# #
# # axes[1].set_xlabel('Sepal length', fontsize=14)
# # axes[1].set_xlabel('Sepal width', fontsize=14)
# # axes[1].set_title('After K-Means clustering')
# #
# # # data visualization - before and after clustering for petal length vs petal width
# # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# # axes[0].scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=50)
# # axes[1].scatter(x['Petal Length'], x['Petal Width'], c=color1[pred_y['Target']], s=50)
# # axes[1].scatter(centroids['Petal Length'], centroids['Petal Width'], c='k', s=70)
# # axes[0].set_xlabel('Petal length', fontsize=14)
# # axes[0].set_ylabel('Petal width', fontsize=14)
# # axes[0].set_title('Before K-Means clustering')
# #
# # axes[1].set_xlabel('Petal length', fontsize=14)
# # axes[1].set_xlabel('Petal width', fontsize=14)
# # axes[1].set_title('After K-Means clustering')
#
# plt.show()
#
# # step 3: cluster validity
# from sklearn.metrics import davies_bouldin_score, silhouette_score
#
# # db index
# print('Davis Bouldin index for cluster =', 3, 'is', davies_bouldin_score(x, km.labels_))
# # choose highest
# # silhouette index gives the average value for all the samples
# print('Silhouette index for clusters=', 3, 'is', silhouette_score(x, km.labels_))
#
# # to generate a range of cluster validity scores
# no_clusters = [2, 3, 4, 5, 6]
# # cluster validity over a range of clusters
# for i in no_clusters:
#     # perform k-means
#     km1 = KMeans(n_clusters=i)
#     clabel = km1.fit_predict(x)
#     # choose the minimum value among all of them for DB
#     # ....... maximum value among all of them for Silhouette index
#
#     # DB index
#     print('Davis Bouldin index for cluster =', i, 'is', davies_bouldin_score(x, clabel))
#     # The silhouette_score gives the average value for all the samples
#     print('Silhouette index for clusters =', i, 'is', silhouette_score(x, clabel))
#
# # hierarchical clustering
# import plotly.figure_factory as ff
# from scipy.cluster import hierarchy
#
# # get the linkage matrix using dissimilarity -euclidean or similarity - cosine
# linkage = hierarchy.linkage(x, metric='euclidean')
#
# # plot the dendrogram
# fig = plt.figure(figsize=(25, 10))
# s = hierarchy.dendrogram(linkage, leaf_font_size=12)
# # plt.show()
# # dissimilarity: larger then dissimilar, smaller then similar
# # similarity (cosine): opposite
#

