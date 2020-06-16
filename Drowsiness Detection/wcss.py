import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

features = np.genfromtxt('data/train_features_5folds.csv', missing_values=0, skip_header=1, delimiter=',', dtype=float)

EAR = features[:, 1]
MAR = features[:, 2]
MOE = features[:, 3]
pupil_circularity = features[:, 4]
n_EAR = features[:, 5]
n_MAR = features[:, 6]
n_MOE = features[:, 7]
n_pupil_circularity = features[:, 8]

ones = np.empty_like(n_pupil_circularity)
X = np.c_[n_pupil_circularity, ones]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for n_pupil_circularity')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(X)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()