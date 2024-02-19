import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 创建训练数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 创建神经网络模型
model = Sequential()

# 添加一层全连接层
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X, y, epochs=1000, verbose=0)

# 创建测试数据
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 进行预测
predictions = model.predict(X_test)

# 打印预测结果
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Class {int(round(pred[0]))}")