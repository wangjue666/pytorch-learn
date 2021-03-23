import torch

x = torch.randn(3, 4, requires_grad=True)
y = torch.randn(3, 4, requires_grad=True)

t = x + y
z = t.sum()

z.backward()
print(y.grad)