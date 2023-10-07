import random
import numpy
import matplotlib.pyplot as plt
import time
inf = 99999
dim=2
T=50
n=10000
z=[]

#样本生成函数
def init():
    for _ in range(0,n):        # 总共多少条数据
      x=[]
      for _ in range(0,dim):   # 每条数据多少维
         rand=random.randint(1,1000)
         x.append(rand)
      w=[]
      w.append(x)
      z.append(w)

#距离函数
def dis(a,b):
    aa=a[0]
    bb=b[0]
    sum=0
    for i in range(0,len(aa)):
      if(i>=len(bb)):break
      sum=sum+(aa[i]-bb[i])*(aa[i]-bb[i])
    return numpy.sqrt(sum)

#层次聚类函数
def neighbor():
    while(1):
      minn=inf
      id1=0
      id2=0
      for i in range(0,len(z)):
         for j in range(0,len(z)):
            if(i==j):continue
            if(minn>=dis(z[i],z[j])):
               minn=dis(z[i],z[j])
               id1=i
               id2=j
      if(minn<=T):
         z[id2].extend(z[id1])
         del(z[id1])  
      else:break 

#结果显示函数
def drawpicture():
       color=['c','b','g','r','m','y','k','w']
       for i in range(0,len(z)):
          xx=[]
          yy=[]
          for j in range(0,len(z[i])):
             xx.append(z[i][j][0])
             yy.append(z[i][j][1])
          plt.scatter(xx,yy,c=color[i])
       plt.savefig("../picture/hierarchical clustering.jpg")   

time1=time.time()
init()
neighbor()
time2=time.time()
print(f"spend total time {time2-time1}s")
drawpicture()