import numpy as np
import matplotlib.pyplot as plt
def f(x,y,z):
    return z

def g(x,y,z):
    return 3*y - 2*z

def FuncArray(x,Y):
    return np.array([f(x,Y[0],Y[1]),g(x,Y[0],Y[1])])

def main():
    y0 = 1
    z0 = 1
    h = 0.1
    error  = []
    Y = np.array([y0,z0])
    for i in range(5):
        Yp = Y + h*FuncArray(i*h,Y)
        Yc = Y + h*FuncArray((i+1)*h,Yp)
        Y = 0.5*(Yp+Yc)
        print(Y,end='')
        print(np.exp((i+1)*h))

    # 提供数据
    # x = [0.1, 0.01, 0.001, 0.0001]
    # # 将 x 和 y 转换为 numpy 数组
    # x = np.array(x)
    # error = np.array(error)
    #
    # # 创建对数坐标图
    # plt.figure()
    # plt.plot(x, error, marker='o', linestyle='-')
    #
    # # 设置 x 轴为对数坐标
    # plt.xscale('log')
    #
    # # 设置 y 轴为对数坐标
    # plt.yscale('log')
    #
    # # 添加标题和标签
    # plt.title('log')
    # plt.xlabel('h')
    # plt.ylabel('error')
    #
    # # 显示图形
    # plt.show()


if __name__ == "__main__":
    main()