import torch
import numpy as np
import torch.nn as nn

x_values = [i for i  in range(11)]
x_train = np.array(x_values, dtype = np.float32)
x_train = x_train.reshape(-1, 1)



y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

# 其实线性回归就是一个不加激活函数的全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


# 指定好参数和损失函数
epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 开始训练模型
for epoch in range(epochs):
    epoch += 1
    # 转行成tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    # 梯度要清零每一次迭代
    optimizer.zero_grad() 

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 返向传播
    loss.backward()

    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


# 测试模型预测结果        

predicted = model(torch.from_numpy(x_train).requires_grad_()).data
print(predicted)

print("end")