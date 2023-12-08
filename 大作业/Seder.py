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
'''G = (D - L)_*U'''
temp = np.linalg.inv(D - L)
G = temp.dot(U)
'''测试收敛条件，p(G) < 1'''
# eigenvalues, _ = np.linalg.eig(G)
# print(abs(eigenvalues))
'''f = (D - L)_*P'''
f = temp.dot(P)

'''开始迭代X = B0*X+f'''
'''初值x = x0'''
x = np.zeros(n)
iterations = 100
for i in range(iterations):
    x = G.dot(x) + f
    print(f"第{i + 1}次迭代结果:", x)

