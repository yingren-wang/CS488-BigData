from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn import decomposition
import scipy.io
import sys

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.cluster import hierarchy

print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)
print(sys.getrecursionlimit())

df2 = scipy.io.loadmat(r'indianR.mat')

x = np.array(df2['X'])
gth = np.array(df2['gth'])
num_rows = np.array(df2['num_rows'])
num_cols = np.array(df2['num_cols'])
num_bands = np.array(df2['num_bands'])
bands, samples = x.shape

print(df2['gth'])

# load ground truth data
gth_mat = scipy.io.loadmat(r'indian_gth.mat')
gth_mat = {i: j for i, j in gth_mat.items() if i[0] != '_'}
gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})

# List features
n = []
ind = []
for i in range(bands):
    n.append(i+1)
for i in range(bands):
    ind.append('band' + str(n[i]))

features = ind

# Normalize the features
scaler_model = MinMaxScaler()
scaler_model.fit(x.astype(float))
x = scaler_model.transform(x)

# -----------------PCA dimension reduction-----------------
# Finding the principle components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
x1 = x.transpose()
x = pd.DataFrame(x1, columns=ind)
x_pca = np.matmul(x1, principalComponents)
x_pca.shape
x_pca = pd.DataFrame(data=x_pca, columns=['PC-1', 'PC-2'])

# -----------------LDA dimension reduction---------------
lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit(x1, gt.values.ravel()).transform(x1)
x_lda.shape
x_lda = pd.DataFrame(data=x_lda, columns=['PC-1', 'PC-2'])

# # -----------------Elbow Method for Indian Pines------------------
# # step 1: pre-clustering
# # k-means clustering
# cost = []
# for i in range(1, 17):
#     km = KMeans(n_clusters=i, max_iter=500)
#     # perform k-means clustering on data X
#     km.fit(x)
#     # calculates squared error for the clustered points
#     cost.append(km.inertia_)
#
# # plot the cost against k values
# plt.plot(range(1, 17), cost, color='b', linewidth='4')
# plt.xlabel("Value of K")
# plt.ylabel("Squared Error (Cost)")
# plt.title("Elbow method Indian Pine")
# plt.show()

# ----------------KMeans without reduction---------------------
km = KMeans(n_clusters=2, n_jobs=3, random_state=111)
km.fit(x)
centroids = pd.DataFrame(km.cluster_centers_, columns=ind)
print(centroids)
# prediction label colors
# prediction label colors
colors = np.array(['b', 'r', 'g', 'y', 'm', 'c', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'k', 'b', 'r', 'g'])
color1 = np.array(['b', 'r', 'g', 'y', 'm', 'c', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'k', 'b', 'r', 'g'])
pred_y = pd.DataFrame(km.labels_, columns=['gth'])

# # data visualization pca - before and after clustering for Sepal length vs Sepal width
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].scatter(x['band1'], x['band2'], c=colors[gt["gth"]], s=50)
# axes[1].scatter(x['band1'], x['band2'], c=colors[pred_y["gth"]], s=50)
# axes[1].scatter(centroids['band1'], centroids['band2'], c='k', s=70)
# axes[0].set_xlabel('band1', fontsize=14)
# axes[0].set_ylabel('band2', fontsize=14)
# axes[0].set_title('Before K-Means clustering Without Dimensionality Reduction')
#
# axes[1].set_xlabel('band1', fontsize=14)
# axes[1].set_ylabel('band2', fontsize=14)
# axes[1].set_title('After K-Means clustering Without Dimensionality Reduction')
# plt.show()

# ----------------PCA KMeans---------------------------
km = KMeans(n_clusters=2, n_jobs=2, random_state=111)
km.fit(x_pca)
centroids = pd.DataFrame(km.cluster_centers_, columns=['PC-1', 'PC-2'])
print(centroids)

# # data visualization pca - before and after clustering for Sepal length vs Sepal width
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].scatter(x_pca['PC-1'], x_pca['PC-2'], c=colors[gt["gth"]], s=50)
# axes[1].scatter(x_pca['PC-1'], x_pca['PC-2'], c=color1[pred_y['gth']], s=50)
# axes[1].scatter(centroids['PC-1'], centroids['PC-2'], c='k', s=70)
# axes[0].set_xlabel('PC-1', fontsize=14)
# axes[0].set_ylabel('PC-2', fontsize=14)
# axes[0].set_title('Before K-Means clustering Indian Pines PCA')
# axes[1].set_xlabel('PC-1', fontsize=14)
# axes[1].set_xlabel('PC-2', fontsize=14)
# axes[1].set_title('After K-Means clustering Indian Pines PCA')
# plt.show()

# --------------------LDA KMeans----------------------------
km = KMeans(n_clusters=2, n_jobs=2, random_state=111)
km.fit(x_lda)
centroids = pd.DataFrame(km.cluster_centers_, columns=['PC-1', 'PC-2'])
print(centroids)
# prediction label colors
pred_y = pd.DataFrame(km.labels_, columns=['gth'])

# # data visualization pca - before and after clustering for Sepal length vs Sepal width
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].scatter(x_lda['PC-1'], x_lda['PC-2'], c=colors[gt["gth"]], s=50)
# axes[1].scatter(x_lda['PC-1'], x_lda['PC-2'], c=color1[pred_y["gth"]], s=50)
# axes[1].scatter(centroids['PC-1'], centroids['PC-2'], c='k', s=70)
# axes[0].set_xlabel('PC-1', fontsize=14)
# axes[0].set_ylabel('PC-2', fontsize=14)
# axes[0].set_title('Before K-Means clustering Indian Pines LDA')
# axes[1].set_xlabel('PC-1', fontsize=14)
# axes[1].set_xlabel('PC-2', fontsize=14)
# axes[1].set_title('After K-Means clustering Indian Pines LDA')
# plt.show()

# # ----------------Hierarchical dissimilarity LDA----------------------
# x_lda1 = x_lda.iloc[0:5256]
# x_lda2 = x_lda.iloc[5257:10513]
# x_lda3 = x_lda.iloc[10514:15769]
# x_lda4 = x_lda.iloc[15770:21025]
# # part 1
# linkage = hierarchy.linkage(x_lda1, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity LDA part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_lda2, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity LDA part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_lda3, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity LDA part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_lda4, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity LDA part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()

# ----------------Hierarchical similarity LDA-------------------------
# x_lda1 = x_lda.iloc[0:5256]
# x_lda2 = x_lda.iloc[5257:10513]
# x_lda3 = x_lda.iloc[10514:15769]
# x_lda4 = x_lda.iloc[15770:21025]
# # part 1
# linkage = hierarchy.linkage(x_lda1, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity LDA part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_lda2, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity LDA part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_lda3, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity LDA part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_lda4, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity LDA part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()
#
# # ----------------Hierarchical dissimilarity PCA----------------------
# x_pca1 = x_pca.iloc[0:5256]
# x_pca2 = x_pca.iloc[5257:10513]
# x_pca3 = x_pca.iloc[10514:15769]
# x_pca4 = x_pca.iloc[15770:21025]
#
# linkage = hierarchy.linkage(x_pca1, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity PCA part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_pca2, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity PCA part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_pca3, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity PCA part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_pca4, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity PCA part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()
#
# # ----------------Hierarchical similarity PCA-------------------------
# linkage = hierarchy.linkage(x_pca1, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity PCA part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_pca2, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity PCA part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_pca3, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity PCA part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_pca4, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity PCA part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()

# ------------------Hierarchical dissimilarity without reduction---------
x_1 = x.iloc[0:5256]
x_2 = x.iloc[5257:10513]
x_3 = x.iloc[10514:15769]
x_4 = x.iloc[15770:21025]

# # part 1
# linkage = hierarchy.linkage(x_1, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity without reduction part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_2, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity without reduction part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_3, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity without reduction part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_4, metric='euclidean')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical Dissimilarity without reduction part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()
#
#
# # ----------------Hierarchical similarity without reduction-------------------------
# # part 1
# linkage = hierarchy.linkage(x_1, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity without reduction part1")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 2
# linkage = hierarchy.linkage(x_2, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity without reduction part2")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 3
# linkage = hierarchy.linkage(x_3, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity without reduction part3")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# # part 4
# linkage = hierarchy.linkage(x_4, metric='cosine')
# fig = plt.figure(figsize=(25, 10))
# plt.title("Hierarchical similarity without reduction part4")
# s = hierarchy.dendrogram(linkage, leaf_font_size=5)
# plt.show()


# ----------------Davies Bouldin and Silhouette index----------
no_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for i in no_clusters:
    # perform k-means
    km1 = KMeans(n_clusters=i)
    clabel_1 = km1.fit_predict(x_1)
    clabel_2 = km1.fit_predict(x_2)
    clabel_3 = km1.fit_predict(x_3)
    clabel_4 = km1.fit_predict(x_4)
    # choose the minimum value among all of them for DB
    # ....... maximum value among all of them for Silhouette index

    # DB index
    davies_bouldin = (davies_bouldin_score(x_1, clabel_1) + davies_bouldin_score(x_2, clabel_2) + davies_bouldin_score(x_3, clabel_3) + davies_bouldin_score(x_4, clabel_4))/4

    print('Davis Bouldin index for cluster =', i, 'is', davies_bouldin)
    # The silhouette_score gives the average value for all the samples
    silhouette = (silhouette_score(x_1, clabel_1) + silhouette_score(x_2, clabel_2) + silhouette_score(x_3, clabel_3) + silhouette_score(x_4, clabel_4))/4
    print('Silhouette index for clusters =', i, 'is', silhouette)

