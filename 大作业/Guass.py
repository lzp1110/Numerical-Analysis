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
    if B[i,i] == 0:
        '''寻找第i列[i+1,-1]行不为0的元素'''
        for row in range(i+1,n_rows):
            if B[row,i] != 0:
                '''交换B矩阵和P矩阵中各自对应的两行元素'''
                B[i] , B[row] = B[row] , B[i]
                P[i], P[row] = P[row], P[i]
                # '''更新sol_index以确定最后输出的解的位置'''
                # sol_index[i], sol_index[column] = sol_index[column], sol_index[i]
                break
    '''完成交换，此时B[i,i] != 0 ,进行消元'''
    '''依次对第[i+1,-1]行进行消元'''
    for row in range(i+1,n_rows):
        '''计算消元乘子'''
        m = B[row,i]/B[i,i]
        '''进行消元，结果保留number位小数'''
        B[row] = np.round(B[row] - m * B[i], number)
        P[row] = np.round(P[row] - m * P[i], number)
    '''至此完成第i次消元'''
# print(B)
# df_B = pd.DataFrame(B)

#
'''从最后一行开始逐行往回进行回代求解，结果保留number位小数'''
for row in np.arange(n_rows)[::-1]:
   sol[row] = round((P[row] - sum(B[row,column]*sol[column] for column in range(row + 1,n_columns)))/B[row,row],number)


endT = time.time()
print(f"求得结果(保留{number}位小数):", sol)
sol = sol.reshape(1,8)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
sol = pd.DataFrame(sol)
sol.to_csv('./runData/Guass/sol.csv',header=header,index=None)

useT = np.array([endT - startT])
print("求解消耗时间:", useT , "ms")
useT = pd.DataFrame(useT)
useT.to_csv('./runData/Guass/time.csv',header=None,index=None,mode='a')




