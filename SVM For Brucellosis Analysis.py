import numpy as np
from sklearn import svm

# 定义样本文件路径
sample_file = 'sample_data.txt'

# 定义批大小和特征维度
batch_size = 100
num_features = 13

# 创建SVM分类器
clf = svm.SVC(kernel='linear')

# 逐批读取样本并进行训练
with open(sample_file, 'r') as f:
    for batch_start in range(0, 500, batch_size):
        # 读取当前批次的样本数据
        batch_data = []
        batch_labels = []
        for line in f.readlines()[batch_start:batch_start+batch_size]:
            values = line.strip().split(',')
            features = [float(x) for x in values[:num_features]]
            label = int(values[-1])
            batch_data.append(features)
            batch_labels.append(label)

        # 将数据转换为NumPy数组
        X_batch = np.array(batch_data)
        y_batch = np.array(batch_labels)

        # 训练当前批次的样本
        clf.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))

# 创建测试数据
X_test = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

# 进行预测
predictions = clf.predict(X_test)

# 打印预测结果
print(f"Sample 1: Class {predictions[0]}")