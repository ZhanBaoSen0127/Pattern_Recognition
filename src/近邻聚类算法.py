import numpy as np
import matplotlib.pyplot as plt
import time


"""
样本数据结构: 二维numpy数组, 每一行即为一个样本
聚类中心数据结构:二维numpy数组, 每一行即为一个聚类中心
分类结果数据结构:二维numpy数组,每一行属于一个类, 行标与聚类中心数组的行标一致
                每一列存储对应样本数组的下标。
"""
np.random.seed(28)
threshold = 250

# 基于欧氏距离
# 返回值：(是否为一个新的聚类中心，聚类中心的下标)
def calcu_dis(centers, sample, threshold) -> list:
    dis_list = []
    result = [False, -1]
    # print(f"centers {centers}")
    # 距离计算
    for center in centers:
        distance = np.sqrt(np.sum((center-sample)**2))
        dis_list.append(distance)
    # 判断是否形成新的聚类中心
    # print(dis_list)
    for dis in dis_list:
        if dis < threshold:
            result[0] = False
            break
        else:
            result[0] = True

    if result[0] == False:
        index = dis_list.index(min(dis_list))
        result[1] = index
    
    return result

# colors = []

num = 0

def draw_picture(classify:dict, sample_array:np.ndarray):
    global num
    center_indexs = [int(x) for x in classify.keys()] 
    # print(f"index {center_indexs}")
    for i in center_indexs:
        plt.scatter(sample_array[i][0], sample_array[i][1], s=200, c='k', marker='X')
    sample_list_total = list(classify.values())
    sample_list_one = [j for i in sample_list_total for j in i]
    try:
        for sample_list in sample_list_total:
            plt.scatter(sample_array[sample_list, 0], sample_array[sample_list, 1], s=60)
            num += 1

    except IndexError as e:
        print(e)


    plt.title('nearest neighbor clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.savefig("近邻聚类算法3.jpg")
    # plt.show()

# 样本生成
sample_array = np.random.randint(low=0, high=1000, size=(1000,2), dtype=np.int32)
np.savetxt('近邻聚类算法数据集3.txt', sample_array)

# 第一个聚类中心选择
first_center_index = np.random.randint(low=0, high=200, size=1, dtype=np.int32)
first_center = sample_array[first_center_index[0]]

# 聚类中心数组初始化   key---代表聚类中心在样本数组中的下标  value---是一个list，代表哪些元素在聚类中心中
center_array = first_center
temp = []
temp.append(center_array)
center_array = temp
classify = {str(first_center_index[0]):[]}

num = 0
start_time = time.time()
for sample in sample_array:
    result = calcu_dis(center_array, sample, threshold)
    if(result[0] == True):  # 若为一个新的聚类中心
        new_center = sample
        # print(f"new center {new_center}")
        center_array = np.vstack((center_array, new_center))
        classify[str(num)] = []
    else:                   # 根据返回的属于哪个类的下标进行处理
        keys = list(classify.keys())
        # print(f"keys {keys} type {type(keys)}")
        key = keys[result[1]]
        classify[key].append(num)
        # pass
    num += 1
end_time = time.time()
print(f"spend time {end_time - start_time} s")
draw_picture(classify, sample_array)