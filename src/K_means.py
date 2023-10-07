import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(40)  #设置种子 保证实验的可重复性

size = (2000,2) # 数据集大小
scope = (1,2000)    # 数据大小
threshold = 250
K = 7
# 计算两个点之间的距离---欧几里得距离
def calcudis(a, b):
    return np.sqrt(np.sum((a-b)**2))

# 创建数据集
def createDataSet(size:tuple, scope:tuple):
    sample_array = np.random.randint(low=scope[0], high=scope[1], size=size, dtype=np.int32)
    return sample_array

# 初始化k个聚类中心
def get_init_center(k:int, dataset:np.ndarray):
    index_array = np.random.randint(low=0, high=size[0], size=k, dtype=np.int32)
    init_center_array = dataset[index_array]
    return index_array, init_center_array

def classify_fun(classify_dict:dict, center_array, dataset):
    for i in classify_dict.keys():
        classify_dict[i] = []

    for data in dataset:
        dis = []
        for center in center_array:
            dis.append(calcudis(data, center))
        min_index = dis.index(min(dis))
        classify_dict[str(min_index)].append(data)
    return classify_dict



def get_new_center(classify_dict:dict)->np.ndarray:
    new_center_list = []
    for value in classify_dict.values():
        new_center = sum(value)/len(value)
        new_center_list.append(new_center)
    return np.array(new_center_list)

def judge_continue(old_center, new_center)->bool:
    for i in range(len(old_center)):
        dis = calcudis(old_center[i], new_center[i])
        if dis > threshold:
            return True
    return False


def drawpicture(center_array:np.ndarray, dataset:np.ndarray, classify_dict:dict):
    colors = plt.cm.viridis(np.linspace(0,1,center_array.shape[0]))
    classify_value = list(classify_dict.values())
    j=0
    for value in classify_value:
        value = np.array(value)
        plt.scatter(value[:,0],value[:,1],s=20,c=[colors[j]])
        j+=1
    for i in range(center_array.shape[0]):
        plt.scatter(center_array[i][0], center_array[i][1], s=200, c='k', marker='X')

    plt.title('K-Means')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.grid(True)
    plt.savefig("../picture/k_means3.jpg")    


if __name__ == "__main__":
    dataset = createDataSet(size, scope)
    index_array, center_array = get_init_center(K, dataset)
    classify_dict = {}
    for i in range(len(index_array)):
        classify_dict[str(i)] = [dataset[index_array[i]]]

    start = time.time()
    classify_dict = classify_fun(classify_dict, center_array, dataset)
    new_center = get_new_center(classify_dict)
    while(judge_continue(center_array, new_center)):
        center_array = new_center
        classify_dict = classify_fun(classify_dict, center_array, dataset)
        new_center = get_new_center(classify_dict)
    end = time.time()
    print(f"spend time {end-start} s \n分类结束!!!")

    drawpicture(new_center, dataset, classify_dict)    