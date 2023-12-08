import numpy as np

from problem import *

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
print(f)
'''开始迭代X = B0*X+f'''
'''初值x = x0'''
x_last = np.zeros(n)
iterations = 150
iterData = []
for i in range(iterations):
    x = B0.dot(x_last) + f
    x_last = x
    '''存储x_last'''
    iterData.append(x_last)
    print(f"第{i + 1}次迭代结果:", x_last)

iterData = np.array(iterData)



'''绘图'''
fig = plt.figure(figsize = (10,5))
x = np.arange(iterations)
'''依次绘制每个解的迭代过程'''
for i in range(8):
    plt.plot(x,iterData[:,i])
plt.xlabel("iteration")
plt.title("Jacobi Iteration")
plt.savefig("./runData/Jacobi/" + f"iteration_{iterations}" + ".png")
plt.show()
