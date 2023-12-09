import numpy as np
import pandas as pd
from pypower.api import case9
# for i in np.arange(8)[::-1]:
#     print(i)

# a = np.array([1,2,3])
# print(list(a).index(2))
# print(type(a))
# 13.295207774491775
# -0.07521507124684451
# -1.0
# # -1.1102230246251565e-16
# print(-0.07521507124684451*13.295207774491775)
# a = list(np.arange(9)).remove(1)
#
# print(1/2.334)
# 导入IEEE 9节点系统
ppc = case9()
# network.import_from_pypower_ppc(ppc)
# 打印系统信息
branch_data = ppc['branch']
# print(branch_data)
# bus = branch_data[:, 0:2]
# X = branch_data[:, 3]
# X = X.reshape(9,1)
#
# data = np.hstack((bus,X))
# data = pd.DataFrame(data)
# data.to_csv("./runData/problem/branchData.csv")

# a = np.array([10.1875, 4.981013548356851, 0.0, -5.372524907622419, 0.0, -4.200004200004201,0.0, -6.953754749414494])
# a = np.round(a,4)

from problem import *

B,P = GetProblem()


'''定义松弛因子w'''
w = 0.0
n = B.shape[0]
'''将B分解为D-L-U'''
'''L为下三角，U为上三角'''
L = -np.tril(B)
U = -np.triu(B)
D = np.zeros((n,n))
I = np.eye(n)
for i in range(n):
    D[i,i] = B[i,i]
    L[i,i] = 0
    U[i,i] = 0

iterData_0 = []

w = 1.6
'''G = (D - L)_*U'''
temp = np.linalg.inv(D - w*L)
G = temp.dot((1-w)*D+w*U)
'''测试收敛条件，p(G) < 1'''
# eigenvalues, _ = np.linalg.eig(G)
# print(abs(eigenvalues))
'''f = (D - L)_*P'''
f = w*temp.dot(P)

'''开始迭代X = G*X+f'''
'''初值x = x0'''
x_last = np.zeros(n)
e = 0.0001
iterData = []

x = G.dot(x_last) + f

iter_num = 1
iterData.append(x)
while max(abs(x - x_last)) >= e:
        x_last = x
        x = G.dot(x_last) + f
        '''存储x'''
        iterData.append(x)
        iter_num += 1
        print(f"第{iter_num}次迭代结果:", x)

iterData = np.array(iterData)
x = x.reshape(1,8)
# x = np.round(x,4)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
x = pd.DataFrame(x)
x.to_csv(f'./runData/SOR/w={w}.csv',header=header,index=None)

#
# '''绘图'''
# fig = plt.figure(figsize = (10,5))
# x = np.arange(iter_num)
# '''依次绘制每个解的迭代过程'''
# for i in range(8):
#     plt.plot(x,iterData[:,i])
#     plt.xlabel("iter_num")
#     plt.title(f"SOR Iteration(w ={w})")
# plt.savefig("./runData/SOR/" + f"iteration_{w}_{iter_num}" + ".png")
