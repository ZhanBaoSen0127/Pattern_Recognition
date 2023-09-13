import numpy as np

# a = [2,4]
# b = np.array([[1,2],[1,2],[2,3],[4,5],[5,6]])
# print(b[a])

a = [[], []]
print(len(a))
# b = np.array([1,2,3,4])
# print(b[a])

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# # 创建示例数据，您可以替换为自己的数据
# n_samples = 300
# n_features = 2
# n_clusters = 3
# random_state = 42
# X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
# print(type(X))
# print(X.shape)
# # 使用K均值聚类算法
# kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
# kmeans.fit(X)

# # 获取聚类中心和标签
# cluster_centers = kmeans.cluster_centers_
# labels = kmeans.labels_

# # 设置颜色映射
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# # 创建散点图
# plt.figure(figsize=(8, 6))

# for i in range(n_clusters):
#     # 绘制每个类的数据点
#     plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], s=50, c=colors[i], label=f'Cluster {i + 1}')

# # 绘制聚类中心
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='k', marker='X', label='Cluster Centers')

# plt.title('K-Means Clustering')
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.grid(True)
# plt.show()
