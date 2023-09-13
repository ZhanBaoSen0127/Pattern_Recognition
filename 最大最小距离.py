import numpy as np
import matplotlib.pyplot as plt
import time

"""
实现两个函数：
    第一个找到所有的聚类中心
    第二个根据聚类中心进行分类
"""

theta = 0.2
sample_array = np.random.randint(low=0, high=1000, size=(5000,2), dtype=np.int32)

def calcudis(a, b):
    return np.sqrt(np.sum((a-b)**2))

def findCenters(sample_array, theta):
    center_list = []
    first_center_index = np.random.randint(low=0, high=20, size=1, dtype=np.int32)
    first_center = sample_array[first_center_index[0]]
    center_list = [first_center]

    # 得到第二个聚类中心
    distance_list = [calcudis(first_center, sample) for sample in sample_array]    
    max_index = distance_list.index(max(distance_list))
    max_distance = max(distance_list)
    threshold = max_distance * theta
    center_list.append(sample_array[max_index])
    
    while True:
        min_dis = [min([calcudis(center, sample) for center in center_list]) for sample in sample_array]
        max_dis = max(min_dis)
        max_min_index = min_dis.index(max_dis)
        if max_dis < threshold:
            print(f"聚类中心寻找结束")
            break
        else:   # 将其置为新的聚类中心
            center_list.append(sample_array[max_min_index])        

    return center_list

# 根据聚类中心进行分类
def classification(center_list, sample_array):
    num = 0
    classify = {}
    for center in center_list:
        classify[str(num)] = []
        num += 1
    for sample in sample_array:
        dis_list = []
        for center in center_list:
            dis = calcudis(center, sample)
            dis_list.append(dis)
        min_index = dis_list.index(min(dis_list))
        classify[str(min_index)].append(sample)
    print(f"分类结束！！！！")
    return classify

def draw_picture(center_list,classify:dict):

    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    classify_value = list(classify.values())
    # 创建一个颜色列表，每一组对应一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(center_list)))
    j=0
    for value in classify_value:
        for i in value:
            plt.scatter(i[0], i[1], s=20, c=[colors[j]])
        j += 1

    for i in range(len(center_list)):
        plt.scatter(center_list[i][0], center_list[i][1], s=200, c='k', marker='X')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.savefig("最大最小距离.jpg")
    plt.show()


findcenter_time_start = time.time()
center_list = findCenters(sample_array, theta)
findcenter_time_end = time.time()
print(f"findcenter spend {findcenter_time_end - findcenter_time_start} s")

classifcation_time_start = time.time()
classify = classification(center_list, sample_array)
classifcation_time_end = time.time()
print(f"classificatin spend {classifcation_time_end - classifcation_time_start} s")

draw_picture_time_start = time.time()
draw_picture(center_list, classify)
draw_picture_time_end = time.time()
print(f"draw picture spend {draw_picture_time_end - draw_picture_time_start} s")