# import numpy as np
# import pandas as pd
# from pypower.api import case9
# # for i in np.arange(8)[::-1]:
# #     print(i)
#
# # a = np.array([1,2,3])
# # print(list(a).index(2))
# # print(type(a))
# # 13.295207774491775
# # -0.07521507124684451
# # -1.0
# # # -1.1102230246251565e-16
# # print(-0.07521507124684451*13.295207774491775)
# # a = list(np.arange(9)).remove(1)
# #
# # print(1/2.334)
# # 导入IEEE 9节点系统
# ppc = case9()
# # network.import_from_pypower_ppc(ppc)
# # 打印系统信息
# branch_data = ppc['branch']
# # print(branch_data)
# # bus = branch_data[:, 0:2]
# # X = branch_data[:, 3]
# # X = X.reshape(9,1)
# #
# # data = np.hstack((bus,X))
# # data = pd.DataFrame(data)
# # data.to_csv("./runData/problem/branchData.csv")
#
# # a = np.array([10.1875, 4.981013548356851, 0.0, -5.372524907622419, 0.0, -4.200004200004201,0.0, -6.953754749414494])
# # a = np.round(a,4)
#
# from problem import *
#
# B,P = GetProblem()
#
#
# '''定义松弛因子w'''
# w = 0.0
# n = B.shape[0]
# '''将B分解为D-L-U'''
# '''L为下三角，U为上三角'''
# L = -np.tril(B)
# U = -np.triu(B)
# D = np.zeros((n,n))
# I = np.eye(n)
# for i in range(n):
#     D[i,i] = B[i,i]
#     L[i,i] = 0
#     U[i,i] = 0
#
# iterData_0 = []
#
# w = 1.6
# '''G = (D - L)_*U'''
# temp = np.linalg.inv(D - w*L)
# G = temp.dot((1-w)*D+w*U)
# '''测试收敛条件，p(G) < 1'''
# # eigenvalues, _ = np.linalg.eig(G)
# # print(abs(eigenvalues))
# '''f = (D - L)_*P'''
# f = w*temp.dot(P)
#
# '''开始迭代X = G*X+f'''
# '''初值x = x0'''
# x_last = np.zeros(n)
# e = 0.0001
# iterData = []
#
# x = G.dot(x_last) + f
#
# iter_num = 1
# iterData.append(x)
# while max(abs(x - x_last)) >= e:
#         x_last = x
#         x = G.dot(x_last) + f
#         '''存储x'''
#         iterData.append(x)
#         iter_num += 1
#         print(f"第{iter_num}次迭代结果:", x)
#
# iterData = np.array(iterData)
# x = x.reshape(1,8)
# # x = np.round(x,4)
# header = ['x1','x2','x3','x4','x5','x6','x7','x8']
# x = pd.DataFrame(x)
# # x.to_csv(f'./runData/SOR/w={w}.csv',header=header,index=None)
#
# #
# # '''绘图'''
# # fig = plt.figure(figsize = (10,5))
# # x = np.arange(iter_num)
# # '''依次绘制每个解的迭代过程'''
# # for i in range(8):
# #     plt.plot(x,iterData[:,i])
# #     plt.xlabel("iter_num")
# #     plt.title(f"SOR Iteration(w ={w})")
# # plt.savefig("./runData/SOR/" + f"iteration_{w}_{iter_num}" + ".png")
import numpy as np

# A = np.array([[0.6,0.5],
#               [0.1,0.3]])
# A_ = np.linalg.inv(A)
#
# print(np.linalg.norm(A_, ord=np.inf))

import numpy as np

# import numpy as np
#
# def cholesky_crout_decomposition(matrix):
#     n = len(matrix)
#     L = np.eye(n)
#     D = np.zeros((n, n))
#
#     for k in range(n):
#         dot_product = np.dot(L[k, :k], L[k, :k])
#         D[k, k] = matrix[k, k] - dot_product
#
#         for i in range(k+1, n):
#             dot_product = np.dot(L[i, :k], L[k, :k])
#             L[i, k] = (matrix[i, k] - dot_product) / D[k, k]
#
#     return L, D
#
# def solve_linear_system_cholesky_crout(matrix, b):
#     L, D = cholesky_crout_decomposition(matrix)
#     T = L.dot(D)
#     print(T.dot(L.T))
#     # 解 L^T y = b
#     y = np.linalg.solve(L.T, b)
#
#     # 解 Dx = y
#     x = np.linalg.solve(D, y)
#
#     # 解 Lx = y
#     x = np.linalg.solve(L, x)
#
#     return x
#
# # 示例
# A = np.array([[4, 2, 0, 0],
#               [2, 5, 2, 0],
#               [0, 2, 6, 1],
#               [0, 0, 1, 3]])
#
# b = np.array([2, 1, 2, 3])
#
# x = solve_linear_system_cholesky_crout(A, b)
#
# print("线性方程组的解:")
# print(x)
from problem import *
import numpy as np
def sqrt_decomposition(A):
    n = A.shape[0]
    T = np.zeros((n,n))
    L = np.eye(n)
    d = np.zeros((n,n))

    d[0,0] = A[0,0]
    T[0,0] = d[0,0]

    '''开始对T矩阵逐行求解'''
    for i in range(1,n):
        for j in range(i):
            T[i,j] = A[i,j] - sum(T[i,k]*L[i,k] for k in range(j))
            '''求出T[i,j],对应一个L[i,j]'''
            L[i,j] = T[i,j]/d[j,j]
        '''求完第i行所有T元素后，开始求d[i,i]'''
        d[i,i] = A[i,i] - sum(T[i,k]*L[i,k] for k in range(i))
        T[i,i] = d[i,i]
    return T, L, d
def cholesky_decomposition(matrix):
    n = len(matrix)
    lower_triangular = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_val = sum(lower_triangular[i, k] ** 2 for k in range(j))
                lower_triangular[i, j] = np.sqrt(matrix[i, i] - sum_val)
            else:
                sum_val = sum(lower_triangular[i, k] * lower_triangular[j, k] for k in range(j))
                lower_triangular[i, j] = (matrix[i, j] - sum_val) / lower_triangular[j, j]

    return lower_triangular

def solve_linear_system_cholesky(matrix, b):
    # Cholesky分解
    L = cholesky_decomposition(matrix)
    print(L.dot(L.T))
    # 解 Ly = b
    y = np.linalg.solve(L, b)

    # 解 L^T x = y
    x = np.linalg.solve(L.T, y)

    return x

# 示例
# A = np.array([[4, 2, 0, 0],
#               [2, 5, 2, 0],
#               [0, 2, 6, 1],
#               [0, 0, 1, 3]])
#
# b = np.array([2, 1, 2, 3])
A,b = GetProblem()

x = solve_linear_system_cholesky(A, b)

print("线性方程组的解:")
print(x)


