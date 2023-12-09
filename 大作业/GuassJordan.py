import time

from problem import *

'''保留小数位数'''
number = 4

B,P = GetProblem()

n_rows = B.shape[0]
n_columns = B.shape[1]
sol = np.zeros(8)  ##记录求解结果

startT = time.time()
for i in range(n_rows):
    '''进行第i+1次消元'''
    if i != n_rows-1:
        '''寻找第i列[i,-1]行最大的元素并记录位置'''
        mainValue = max(B[i:-1,i])
        mainIndex = list(B[i:-1,i]).index(mainValue) + i
        '''交换B矩阵和P矩阵中各自对应的两行元素'''
        B[i] , B[mainIndex] = B[mainIndex] , B[i]
        P[i], P[mainIndex] = P[mainIndex], P[i]
        # '''更新sol_index以确定最后输出的解的位置'''
        # sol_index[i], sol_index[column] = sol_index[column], sol_index[i]
    '''完成交换，此时B[i,i] == maxValue ,进行消元'''
    '''依次对第除i行外所有行进行消元然后对第i行化1'''
    index = list(np.arange(n_rows))
    index.remove(i)
    for row in index:
        '''计算消元乘子'''
        m = B[row,i]/B[i,i]
        '''进行消元，结果保留number位小数'''
        B[row] = np.round(B[row] - m * B[i], number)
        P[row] = np.round(P[row] - m * P[i], number)
    '''对第i行化1，结果保留number位小数'''
    m = 1 / B[i,i]
    P[i] = np.round(P[i] * m, number)
    B[i] = np.round(B[i] * m, number)
    '''至此完成第i次消元与化1'''
sol = P

endT = time.time()
print(f"求得结果(保留{number}位小数):", sol)
sol = sol.reshape(1,8)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
sol = pd.DataFrame(sol)
sol.to_csv('./runData/GuassJordan/sol.csv',header=header,index=None)

useT = np.array([endT - startT])
print("求解消耗时间:", useT , "ms")
useT = pd.DataFrame(useT)
useT.to_csv('./runData/GuassJordan/time.csv',header=None,index=None,mode='a')













