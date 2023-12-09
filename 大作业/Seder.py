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

'''开始迭代X = G*X+f'''
'''初值x = x0'''
x_last = np.zeros(n)
e = 0.0001
iterData = []

x = G.dot(x_last) + f


iter_num = 1
iterData.append(x)
while max(abs(x - x_last)) >= e:
    x_last = x
    x = G.dot(x_last) + f
    '''存储x'''
    iterData.append(x)
    iter_num += 1
    print(f"第{iter_num}次迭代结果:", x)

iterData = np.array(iterData)
'''结果写入csv'''
x = x.reshape(1,8)
header = ['x1','x2','x3','x4','x5','x6','x7','x8']
sol = pd.DataFrame(x)
sol.to_csv('./runData/Seidel/sol.csv',header=header,index=None)


'''绘图'''
fig = plt.figure(figsize = (10,5))
x = np.arange(iter_num)
'''依次绘制每个解的迭代过程'''
for i in range(8):
    plt.plot(x,iterData[:,i])
plt.xlabel("iter_num")
plt.title("Guass-Seidel Iteration")
plt.savefig("./runData/Seidel/" + f"iteration_{iter_num}" + ".png")
plt.show()


