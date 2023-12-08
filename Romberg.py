import numpy as np

def func(x):
    return 1/x

def divide_2(p_list):
    c_list = list(np.copy(p_list))
    for i in range(len(p_list)-1):
        new_value = (p_list[i] + p_list[i+1])/2
        c_list.insert(2*i+1,new_value)
    return c_list
#
def T(x_list):
    y = 0
    h = x_list[1] - x_list[0]
    for i in range(len(x_list)-1):
        y += (func(x_list[i]) + func(x_list[i+1]))*h/2
    return y

print("---------开始计算T表--------")
curr_list = [1,3]
T_list = []
value = [T(curr_list)]
T_list.append(value)
iteration = 0
while iteration == 0 or abs(T_list[iteration][-1] - T_list[iteration-1][-1]) > 0.0001:
    iteration += 1
    value = []
    curr_list = divide_2(curr_list)  ##对其进行二分
    value.append(T(curr_list))       ##每一行的第一个值
    for i in range(1,iteration+1):
        value.append((4**i/(4**i-1)*value[i-1] - (1/(4**i-1))*T_list[iteration-1][i-1]))
    T_list.append(value)

print("加速次数：", iteration)

for i in range(len(T_list)):
   print(T_list[i])