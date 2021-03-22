import torch

# 随机矩阵
x = torch.rand(5, 3)
print(x)


# 全0矩阵
# size() 相当于 shape
zero = torch.zeros(5, 3,)
print(zero, zero.size())

zero = torch.add(x, zero)
print(zero)

# numpy的reshape操作
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x, '\n',y, '\n', z)