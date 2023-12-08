from problem import *

'''保留小数位数'''
number = 4

B, P = GetProblem()


def lu_decomposition(A):
    n = A.shape[0]
    U = np.zeros((n,n))
    L = np.eye(n)
    '''U的第一行就是A的第一行'''
    for i in range(n):
        U[0,i] = A[0,i]
    '''L的第一列'''
    for i in range(1,n):
        L[i,0] = round(A[i,0]/U[0,0], number)

    for r in range(1,n):
        '''计算U第r行元素'''
        for i in range(r,n):
            U[r,i] = round(A[r,i] - sum(L[r,k]*U[k,i] for k in range(0,r)), number)
        if r != n-1:
            '''计算L第r列元素'''
            for i in range(r+1,n):
                L[i,r] = round((A[i,r] - sum(L[i,k]*U[k,r] for k in range(0,r)))/U[r,r], number)

    return L,U


startT = time.time()
L,U = lu_decomposition(B)
# print(L)
# print(U)
n = L.shape[0]

'''LUX = P'''
'''先解LY=P'''
Y = np.zeros((n))
'''从最第一行开始逐行往下进行回代求解'''
for row in np.arange(n):
   Y[row] = round((P[row] - sum(L[row,column]*Y[column] for column in range(0,row)))/L[row,row], number)
'''再解UX = Y'''
'''从最后一行开始逐行往回进行回代求解'''
X = np.zeros((n))
for row in np.arange(n)[::-1]:
   X[row] = round((Y[row] - sum(U[row,column]*X[column] for column in range(row + 1,n)))/U[row,row], number)
endT = time.time()

print(f"求得结果(保留{number}位小数):", X)
X = pd.DataFrame(X)
X.to_csv('./runData/LU/LU_sol.csv',header=None)
print("求解消耗时间:", endT - startT , "ms")








