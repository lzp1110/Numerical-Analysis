import numpy as np

from problem import *

'''保留小数位数'''
number = 4

B, P = GetProblem()

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
            T[i,j] = A[i,j] - sum(T[i,k]*L[j,k] for k in range(j))
            '''求出T[i,j],对应一个L[i,j]'''
            L[i,j] = T[i,j]/d[j,j]
        '''求完第i行所有T元素后，开始求d[i,i]'''
        d[i,i] = A[i,i] - sum(T[i,k]*L[i,k] for k in range(i))
        T[i,i] = d[i,i]
    return T, L






startT = time.time()
T,L = sqrt_decomposition(B)
L_t = L.T

n = L.shape[0]
'''T L_t x = P'''
'''先解TY=P'''
Y = np.zeros((n))
'''从最第一行开始逐行往下进行回代求解'''
for row in np.arange(n):
   Y[row] = round((P[row] - sum(T[row,column]*Y[column] for column in range(0,row)))/T[row,row], number)
'''再解LX = Y'''
'''从最后一行开始逐行往回进行回代求解'''
X = np.zeros((n))
for row in np.arange(n)[::-1]:
   X[row] = round((Y[row] - sum(L_t[row,column]*X[column] for column in range(row + 1,n)))/L_t[row,row], number)
endT = time.time()
#
print(f"求得结果(保留{number}位小数):", X)
X = X.reshape(1,8)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
X = pd.DataFrame(X)
X.to_csv('./runData/sqrt/sqrt_sol.csv',header=header,index=None)
#
#
useT = np.array([endT - startT])
print("求解消耗时间:", useT , "ms")
useT = pd.DataFrame(useT)
useT.to_csv('./runData/sqrt/time.csv',header=None,index=None,mode='a')








