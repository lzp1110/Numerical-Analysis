import matplotlib.pyplot as plt

from problem import *

B,P = GetProblem()


'''定义松弛因子w'''
w = 0.0
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


'''低松弛迭代'''
fig1 = plt.figure(figsize = (15,15))
iterData_0 = []
for k in range(9):
    w = round(w + 0.1, 1)
    '''G = (D - L)_*U'''
    temp = np.linalg.inv(D - w*L)
    G = temp.dot((1-w)*D+w*U)
    '''测试收敛条件，p(G) < 1'''
    # eigenvalues, _ = np.linalg.eig(G)
    # print(abs(eigenvalues))
    '''f = (D - L)_*P'''
    f = w*temp.dot(P)

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
        # print(f"第{iter_num}次迭代结果:", x)

    iterData = np.array(iterData)
    iterData_0.append(iterData[:,0])



    '''绘图'''
    ax = fig1.add_subplot(3,3,k+1)
    x = np.arange(iter_num)
    '''依次绘制每个解的迭代过程'''
    for i in range(8):
        ax.plot(x,iterData[:,i])
    # ax.set_xlabel("iter_num",fontsize=6)
    ax.set_title(f"SOR Iteration(w ={w})",fontsize=10)

    print(f"w={w}迭代完成")
    # plt.show()
plt.savefig("./runData/SOR/" + f"iteration_low" + ".png")

'''高松弛迭代'''
fig2 = plt.figure(figsize = (15,15))
w = 1.0
for k in range(9):
    w = round(w + 0.1, 1)
    '''G = (D - L)_*U'''
    temp = np.linalg.inv(D - w*L)
    G = temp.dot((1-w)*D+w*U)
    '''测试收敛条件，p(G) < 1'''
    # eigenvalues, _ = np.linalg.eig(G)
    # print(abs(eigenvalues))
    '''f = (D - L)_*P'''
    f = w*temp.dot(P)

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
        # print(f"第{iter_num}次迭代结果:", x)

    iterData = np.array(iterData)
    iterData_0.append(iterData[:,0])



    '''绘图'''
    ax = fig2.add_subplot(3,3,k+1)
    x = np.arange(iter_num)
    '''依次绘制每个解的迭代过程'''
    for i in range(8):
        ax.plot(x,iterData[:,i])
    # ax.set_xlabel("iter_num",fontsize=6)
    ax.set_title(f"SOR Iteration(w ={w})",fontsize=10)

    print(f"w={w}迭代完成")
    # plt.show()
plt.savefig("./runData/SOR/" + f"iteration_high" + ".png")


'''在同一张图上绘制不同w'''
fig = plt.figure(figsize = (10,5))
for data in iterData_0:
    plt.plot(range(len(data)),data)
plt.xlabel("iter_num")
plt.title("Comparison of Different W")
plt.show()
