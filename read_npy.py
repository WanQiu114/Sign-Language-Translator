import numpy as np


file_path = 'prediction.npy'

# 读取 .npy 文件
data = np.load(file_path)

# 打印数据
print(data)