import numpy as np
import pandas as pd
from pypower.api import case9
# for i in np.arange(8)[::-1]:
#     print(i)

# a = np.array([1,2,3])
# print(list(a).index(2))
# print(type(a))
# 13.295207774491775
# -0.07521507124684451
# -1.0
# # -1.1102230246251565e-16
# print(-0.07521507124684451*13.295207774491775)
# a = list(np.arange(9)).remove(1)
#
# print(1/2.334)
# 导入IEEE 9节点系统
ppc = case9()
# network.import_from_pypower_ppc(ppc)
# 打印系统信息
branch_data = ppc['branch']
# print(branch_data)
# bus = branch_data[:, 0:2]
# X = branch_data[:, 3]
# X = X.reshape(9,1)
#
# data = np.hstack((bus,X))
# data = pd.DataFrame(data)
# data.to_csv("./runData/problem/branchData.csv")