import numpy as np
from scipy.stats import chi2_contingency

A = 3
B = 1
C = 2
D = 0
E = 23 - A
F = 60 - B
G = 211 - C
H = 97 - D

# 创建一个2x2的列联表（可根据实际情况进行修改）
observed = np.array([[A, B, C, D],
                     [E, F, G, H]])

# 进行卡方检验
chi2, p_value, _, _ = chi2_contingency(observed)

# 输出结果
print("卡方统计量: ", chi2)
print("p-value: ", p_value)