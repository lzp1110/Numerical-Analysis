import time

from problem import *

'''保留小数位数'''
number = 4


B,P = GetProblem()

n_rows = B.shape[0]
n_columns = B.shape[1]
sol = np.zeros(8)  ##记录求解结果
startT = time.time()
'''进行第i次消元'''
for i in range(n_rows-1):
    '''寻找第i列[i,-1]行最大的元素并记录位置'''
    mainValue = max(B[i:-1,i])
    mainIndex = list(B[i:-1,i]).index(mainValue) + i
    '''交换B矩阵和P矩阵中各自对应的两行元素'''
    B[i], B[mainIndex] = B[mainIndex], B[i]
    P[i], P[mainIndex] = P[mainIndex], P[i]
        # '''更新sol_index以确定最后输出的解的位置'''
        # sol_index[i], sol_index[column] = sol_index[column], sol_index[i]
    '''完成交换，此时B[i,i] == maxValue ,进行消元'''
    '''依次对第[i+1,-1]行进行消元'''
    for row in range(i+1,n_rows):
        '''计算消元乘子'''
        m = B[row,i]/B[i,i]
        '''进行消元，结果保留number位小数'''
        B[row] = np.round(B[row] - m * B[i], number)
        P[row] = np.round(P[row] - m * P[i], number)
    '''至此完成第i次消元'''
    # df_B = pd.DataFrame(B)
    # df_P = pd.DataFrame(P)
    # df_B.to_csv(f'./cGuass/cGuass_{i + 1}.csv',header=None,index=None)
    # df_P.to_csv(f'./cGuass/cGuass_{i + 1}.csv', header=None, index=None,mode='a')

'''从最后一行开始逐行往回进行回代求解，结果保留number位小数'''
for row in np.arange(n_rows)[::-1]:
   sol[row] = round((P[row] - sum(B[row,column]*sol[column] for column in range(row + 1,n_columns)))/B[row,row],number)


endT = time.time()
print(f"求得结果(保留{number}位小数):", sol)
sol = pd.DataFrame(sol)
sol.to_csv('./runData/cGuass/sol.csv',header=None)
print("求解消耗时间:", endT - startT , "ms")





