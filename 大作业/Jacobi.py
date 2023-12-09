
from problem import *

'''保留小数位数'''
number = 4

B,P = GetProblem()
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
'''B0 = D_*(L+U)'''
D_ = np.linalg.inv(D)
B0 = D_.dot(L+U)
'''测试收敛条件，p(B0) < 1'''
# eigenvalues, _ = np.linalg.eig(B0)
# print(abs(eigenvalues))
f = D_.dot(P)


'''开始迭代X = B0*X+f'''
'''初值x = x0'''
x_last = np.zeros(n)
e = 0.0001
iterData = []

x = B0.dot(x_last) + f


iter_num = 1
iterData.append(x)
while max(abs(x - x_last)) >= e:
    x_last = x
    x = B0.dot(x_last) + f
    '''存储x'''
    iterData.append(x)
    iter_num += 1
    print(f"第{iter_num}次迭代结果:", x)

iterData = np.array(iterData)
'''结果写入csv'''
x = x.reshape(1,8)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
sol = pd.DataFrame(x)
sol.to_csv('./runData/Jacobi/sol.csv',header=header,index=None)


'''绘图'''
fig = plt.figure(figsize = (10,5))
x = np.arange(iter_num)
'''依次绘制每个解的迭代过程'''
for i in range(8):
    plt.plot(x,iterData[:,i])
plt.xlabel("iter_num")
plt.title("Jacobi Iteration")
plt.savefig("./runData/Jacobi/" + f"iteration_{iter_num}" + ".png")
plt.show()
