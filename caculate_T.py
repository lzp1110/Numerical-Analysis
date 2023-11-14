import numpy as np
import sympy as sp
import matplotlib.pyplot as plt



t = [0,5,10,15,20,25,30,35,40,45,50,55]
y = [0,1.27,2.16,2.86,3.44,3.87,4.15,4.37,4.51,4.58,4.62,4.64]

A = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        cishu = i+j
        for k in t:
          A[i,j]+= k**cishu
print(A)

B = np.zeros(3)
for i in range(3):
    for j in range(12):
        B[i] += y[j]*0.0001*(t[j]**i)

print(B)

A_ = np.linalg.inv(A)
X = A_.dot(B)
print(X)
x = np.arange(0,60,5)
y_jinsi = X[0]+X[1]*x+X[2]*x*x

y_true = np.array(y)*0.0001
plt.plot(x,y_jinsi)
plt.plot(x,y_true)
plt.show()
# def func(x):
#     return np.sqrt(1-((972.5/7782.5)**2)*np.sin(x)*np.sin(x))
#     # if x == 0:
#     #     return 1
#     # else:
#     #     return np.sin(x)/x
#
# def divide_2(p_list):
#     c_list = list(np.copy(p_list))
#     for i in range(len(p_list)-1):
#         new_value = (p_list[i] + p_list[i+1])/2
#         c_list.insert(2*i+1,new_value)
#     return c_list
#
# def T(x_list):
#     y = 0
#     h = x_list[1] - x_list[0]
#     for i in range(len(x_list)-1):
#         y += (func(x_list[i]) + func(x_list[i+1]))*h/2
#     return y
# #三个点为一组求解  [0,1,2,3,4]
# def S(x_list):
#     y = 0
#     h = x_list[2] - x_list[0]
#     for i in range(0,len(x_list)-1,2):
#         y += (func(x_list[i]) + 4*func(x_list[i+1]) +func(x_list[i+2]))*h/6
#     return y
# ##五个点为一组求解  [1,2,3,4,5,6,7,8,9]
# def C(x_list):
#     y = 0
#     h = x_list[4] - x_list[0]
#     for i in range(0,len(x_list)-1,4):
#         y += (7*func(x_list[i]) + 32*func(x_list[i+1])+12*func(x_list[i+2])+32*func(x_list[i+3])+7*func(x_list[i+4]))*h/90
#     return y
#
# print("-----------计算实际积分值----------")
# pi = np.pi
# x = sp.symbols('x')
# a = sp.sqrt(1-((972.5/7782.5)**2)*sp.sin(x)*sp.sin(x))
# integrate = sp.integrate(a,(x,0,pi/2))
# print(integrate)
#
# print("---------开始计算--------")
# init_list = [0,pi/2]
#
# T_n = T(init_list)
# d1_list = divide_2(init_list)  ##第1次二分[0,pi/2,pi]
# T_2n = T(d1_list)
# print("T_n:",T_n)
# print("T_2n:",T_2n)
#
# S_n = 4/3*T_2n - 1/3*T_n
# d2_list = divide_2(d1_list)    ##第2次二分
# S_2n = S(d2_list)
# print("S_n:",S_n)
# print("S_2n:",S_2n)
#
# C_n = 16/15*S_2n - 1/15*S_n
# d3_list =divide_2(d2_list)
# C_2n = C(d3_list)
# print("C_n:",C_n)
# print("C_2n:",C_2n)
#
# R_n = 64/63*C_2n - 1/63*C_n
# print("R_n:",R_n)
# # print("相对误差：" , abs((R_n-integrate)/integrate))
#
#
# print("---------开始计算T表--------")
# curr_list = [0,pi/2]
# T_list = []
# value = [T(curr_list)]
# T_list.append(value)
# iteration = 0
# while iteration < 3:
#     iteration += 1
#     value = []
#     curr_list = divide_2(curr_list)  ##对其进行二分
#     value.append(T(curr_list))       ##每一行的第一个值
#     for i in range(1,iteration+1):
#         value.append((4**i/(4**i-1)*value[i-1] - (1/(4**i-1))*T_list[iteration-1][i-1]))
#     T_list.append(value)
#
# print("加速次数：", iteration)
#
# for i in range(len(T_list)):
#    print(T_list[i])