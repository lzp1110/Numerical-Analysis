import numpy as np
import pandas as pd
from pypower.api import case9
import sympy as sp
import time
import matplotlib.pyplot as plt

def is_singular(matrix):
    # 计算矩阵的行列式
    det = np.linalg.det(matrix)
    print(det)
    # 如果行列式为零，则矩阵是奇异矩阵
    if np.isclose(det, 0):
        return True
    else:
        return False

def GetProblem():
    # 导入IEEE 9节点系统
    ppc = case9()
    # network.import_from_pypower_ppc(ppc)
    # 打印系统信息
    branch_data = ppc['branch']

    X = np.zeros((9,9))
    for data in branch_data:
        X[int(data[0])-1,int(data[1])-1] = X[int(data[1])-1,int(data[0])-1] = data[3]
    B = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            if i == j:
                B[i,j] = sum(1/X[i][k] for k in range(9) if X[i][k] != 0)
            if i != j:
                B[i,j] = 0 if X[i,j] == 0 else -1/X[i,j]
    B = B[1:,1:]
    B = np.round(B, 4)
    P = np.array([None, 163, 85, 0, -90, 0, -100, 0, -125])
    P = P[1:]
    x = sp.symbols('x')
    B_ = np.linalg.inv(B)
    # print(B_.dot(P))
    return B,P

if __name__ == "__main__":
    B,P = GetProblem()
    print(P)
    P = P.reshape(P.shape[0],1)
    # problem = np.hstack((B,P))
    # problem = pd.DataFrame(problem)
    # problem.to_csv('./runData/problem/problem_4.csv',index=None)



