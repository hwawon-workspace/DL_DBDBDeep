#%%
import torch
#%%
# 4.2 행렬 곱

x = torch.FloatTensor([[1,2],
                       [3,4],
                       [5,6]])
y = torch.FloatTensor([[1,2],
                       [1,2]])
print(x.size(), y.size())

z = torch.matmul(x,y)
print(z.size())

x = torch.FloatTensor(3,3,2)
y = torch.FloatTensor(3,2,3)

z = torch.bmm(x,y)
print(z.size())
#%%
# 4.4 선형 계층

W = torch.FloatTensor([[1,2],
                       [3,4],
                       [5,6]])
b = torch.FloatTensor([2,2])

def linear(x, W, b):
  y = torch.matmul(x,W) + b
  return y

x = torch.FloatTensor(4,3)
y = linear(x, W, b)
print(y.size())
#%%
# torch.nn.Module 클래스 상속 받기

import torch.nn as nn

class MyLinear(nn.Module):
  def __init__(self, input_dim = 3, output_dim = 2):
    self.input_dim = input_dim
    self.output_dim = output_dim

    super().__init__()

    self.W = torch.FloatTensor(input_dim, output_dim)
    self.b = torch.FloatTensor(output_dim)


  def forward(self, x):
    y = torch.matmul(x, self.W) + self.b
    return y

linear = MyLinear(3,2)
y = linear(x)

for p in linear.parameters():
  print(p)
#%%
# 올바른 방법: nn.Parameter 활용하기

class MyLinear(nn.Module):

  def __init__(self, input_dim = 3, output_dim = 2):
    self.input_dim = input_dim
    self.ouput_dim = output_dim

    super().__init__()

    self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
    self.b = nn.Parameter(torch.FloatTensor(output_dim))

  def forward(self, x):
    y = torch.matmul(x, self.W) + self.b
    return y

#%%
# nn.Linear 활용하기

linear = nn.Linear(3,2)
y = linear(x)

for p in linear.parameters():
  print(p)

class MyLinear(nn.Module):

  def __init__(self, input_dim = 3, output_dim = 2):
    self.input_dim = input_dim
    self.output_dim = output_dim

    super().__init__()

    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    y = self.linear(x)
    return y
#%%
# 5.2 MSE Loss

def mse(x_hat, x):
  y = ((x - x_hat)**2).mean()
  return y

x = torch.FloatTensor([[1,1],
                       [2,2]])
x_hat = torch.FloatTensor([[0,0],
                           [0,0]])
print(mse(x_hat, x))

import torch.nn.functional as F
F.mse_loss(x_hat, x)

F.mse_loss(x_hat, x, reduction = 'sum')

F.mse_loss(x_hat, x, reduction  = 'none')

import torch.nn as nn
mse_loss = nn.MSELoss()
mse_loss(x_hat, x)
#%%
# 6.5 경사하강법 구현

import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.1, .2, .3],
                            [.4, .5, .6],
                            [.7, .8, .9]])

x = torch.rand_like(target)
x.requires_grad = True
print(x)

loss = F.mse_loss(x, target)
loss

threshold = 1e-5
learning_rate = 1.
iter_cnt = 0

while loss > threshold:
  iter_cnt += 1
  loss.backward()

  x = x - learning_rate * x.grad

  x.detach_()
  x.requires_grad_(True)

  loss = F.mse_loss(x, target)

  print('%d-th Loss: %.4e' % (iter_cnt, loss))
  print(x)
#%%
# 6.6 파이토치 오토그래드 소개

x = torch.FloatTensor([[1,2],
                       [3,4]]).requires_grad_(True)

x1 = x + 2
print(x1)

x2 = x - 2
print(x2)

x3 = x1 * x2
print(x3)

y = x3.sum()
print(y)

y.backward()

print(x.grad)

x3.detach_()
# %%
