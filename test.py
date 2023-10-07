import random
import numpy
import matplotlib.pyplot as plt
import time
inf = 99999
dim=2
T=50
n=10000
z=[]
a=[]
k=7
old=[]
new=[]


#样本生成函数
def init():
    for i in range(0,n):
      x=[]
      for j in range(0,dim):
         rand=random.randint(1,100)
         x.append(rand)
      a.append(x)


#距离函数
def dis(aa,bb):
    sum=0
    for i in range(0,len(aa)):
      if(i>=len(bb)):break
      sum=sum+(aa[i]-bb[i])*(aa[i]-bb[i])
    return numpy.sqrt(sum)


#k-均值聚类函数
def neighbor():
    global old
    for i in range(0,k):
       w=[]
       z.append(w)
       new.append(a[i])
       old.append(a[i])

    while(1):
      old.clear()
      old=new.copy()
      for i in range(0,n):
         min=inf
         id=0
         for j in range(0,len(z)):
            if min>=dis(a[i],old[j]):
               min=dis(a[i],old[j])
               id=j
         if(id<len(z)):
            z[id].append(a[i])
      for i in range(0,len(z)):
         sum1=0
         sum2=0
         for j in range(0,len(z[i])):
            sum1+=z[i][j][0]
            sum2+=z[i][j][1]
         if(len(z[i])!=0):
            new[i]=[float(sum1/len(z[i])),float(sum2/len(z[i]))]
      if old==new:break
      else:
         for i in range(0,k):
            z[i].clear()

#结果显示函数
def printf():
       color=plt.cm.viridis(numpy.linspace(0,1,len(z)))
       for i in range(0,len(z)):
          xx=[]
          yy=[]
          for j in range(0,len(z[i])):
             xx.append(z[i][j][0])
             yy.append(z[i][j][1])
          plt.scatter(xx,yy,c=color[i])
          plt.rcParams['font.family'] = ['sans-serif']
          plt.rcParams['font.sans-serif'] = ['SimHei']
          plt.xlabel('第一维')
          plt.ylabel('第二维')
       plt.show()

time1=time.time()
init()
neighbor()
time2=time.time()
print(time2-time1)
printf()



# import numpy as np

# dicts = {'1':2, '2':3}
# for i in dicts:
#     print(dicts[i])

# a = np.array(3)
# if a > 10:
#     print("ddd")
# else:
#     print("aaa")
# print(a)

# dicts = {'0':[np.array([1,1,1]), np.array([1,1,1])],'2':[2,3,4]}
# for value in dicts.values():
#     a = sum(value)/len(value)
#     print(a)


# c = [-100,-100,-5,0,5,3,10,15,-20,25]

# print(c.index(min(c)) ) # 返回最小值
# print(c.index(max(c))) # 返回最大值




# # 假设有两个不同长度的NumPy数组
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6, 7, 8])

# # 找到两个数组的最大长度
# max_length = max(len(array1), len(array2))

# # 创建一个新数组，用0填充
# new_array1 = np.pad(array1, ( max_length - len(array1),0), 'constant')
# new_array2 = np.pad(array2, (0, max_length - len(array2)), 'constant')

# # 将新数组放入一个列表中
# array_list = [new_array1, new_array2]

# # 将列表转换为NumPy数组
# result_array = np.array(array_list)
# print(np.argmax(result_array))
# print(result_array)


# import numpy as np

# # 已知的主对角线元素
# diag_values = np.array([1, 2, 3, 4, 5])

# # 构建对称矩阵
# n = len(diag_values)  # 矩阵的阶数
# symmetric_matrix = np.zeros((n, n))

# # 填充主对角线元素
# np.fill_diagonal(symmetric_matrix, diag_values)

# # 填充对称元素（根据主对角线元素）
# for i in range(n):
#     for j in range(i + 1, n):
#         symmetric_matrix[i, j] = symmetric_matrix[j, i]

# print(symmetric_matrix)


# a = [2,4]
# b = np.array([[1,2],[1,2],[2,3],[4,5],[5,6]])
# print(b[a])

# a = [[], []]
# print(len(a))
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
