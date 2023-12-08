from problem import *

B,P = GetProblem()

'''检查矩阵是否为良性矩阵'''
B_ = np.linalg.inv(B)
det_B = np.linalg.det(B)
det_B_ = np.linalg.det(B_)
print(det_B_*det_B)

def cal_wuqiongfanshu(A):
    row_sum = []
    for i in range(A.shape[0]):
        row_sum.append(sum(abs(A[i,j]) for j in range(A.shape[1])))
    return max(row_sum)

print(cal_wuqiongfanshu(B)*cal_wuqiongfanshu(B_))