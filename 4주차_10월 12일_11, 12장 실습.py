# 11.5 아담 옵티마이저 적용하기
# 기존 캘리포니아 주택 데이터셋에서 SGD -> Adam 으로 교체한 것
# 대부분의 코드 동일하며, 옵티마이저 선언 부분만 달라짐
#%% 1. 데이터 준비
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

df = pd.DataFrame(california.data, columns = california.feature_names)
df['Target'] = california.target
df.tail()

scaler = StandardScaler()
scaler.fit(df.values[:, :-1])
df.values[:, :-1] = scaler.transform(df.values[:, :-1])

df.tail()
#%% 2. 학습 코드 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()

# 입력 샘플: 8차원 벡터, 출력 샘플: 스칼라
x = data[:, :-1]
y = data[:, -1:]

print(x.shape, y.shape)

n_epochs = 4000
batch_size = 256
print_interval = 200
# learning_rate = 1e-2 (이제 학습률이 필요 없게 됨)

# 같은 모델
model = nn.Sequential(
    nn.Linear(x.size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6,5),
    nn.LeakyReLU(),
    nn.Linear(5,4),
    nn.LeakyReLU(),
    nn.Linear(4,3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1))
)

# optim.SGD 썼었는데 optim.Adam으로
# We don't need learning rate hyper-parameter
# lr 인자를 입력으로 받을 수 있지만 웬만해서 튜닝 필요 없음
# 자연어 처리/ 영상처리/ 음성인식 등 심화 과정에서 쓰이는 트랜스포머 사용 수준 다다르면 필요성 생길 것.
optimizer = optim.Adam(model.parameters())

# 이중 for-loop. Adam 옵티마이저 도입으로써 바뀐 부분 없음.
for i in range(n_epochs):
    indices = torch.randperm(x.size(0))
    x_ = torch.index_select(x, dim=0, index = indices)
    y_ = torch.index_select(y, dim=0, index = indices)

    x_ = x_.split(batch_size, dim = 0)
    y_ = y_.split(batch_size, dim = 0)

    y_hat = []
    total_loss = 0

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += float(loss)
        y_hat += [y_hat_i]

    total_loss = total_loss / len(x_)
    if (i+1)% print_interval == 0:
        print('Epoch %d: loss=%.4e' %(i+1, total_loss))

y_hat = torch.cat(y_hat, dim = 0)
y = torch.cat(y_, dim = 0)
#%% 3. 결과 확인
df = pd.DataFrame(torch.cat([y, y_hat], dim = 1).detach().numpy(),
                  columns = ['y', 'y_hat'])
sns.pairplot(df, height = 5)
plt.show()
# %%
# 12.4 데이터 나누기
#%% 1. 데이터 준비
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns = california.feature_names)
df['Target'] = california.target

import torch
import torch.nn as nn
import torch.nn.functional as F

data = torch.from_numpy(df.values).float()

x = data[:, :-1]
y = data[:, -1:]

print(x.size(), y.size())

#Train / Valid / Test Ratio
ratios = [.6, .2, .2]

train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]

print('Train %d / Valid %d / Test %d samples.' % (train_cnt, valid_cnt, test_cnt))

# Shuffle before split
indices = torch.randperm(data.size(0))
x = torch.index_select(x, dim = 0, index = indices)
y = torch.index_select(y, dim = 0, index = indices)

# Split train, valid and test set with each count.
x = list(x.split(cnts, dim = 0))
y = y.split(cnts, dim = 0)

for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())

scaler = StandardScaler()
scaler.fit(x[0].numpy()) # You must fit with train data only.

x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()
#%% 2. 학습 코드 구현
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6,5),
    nn.LeakyReLU(),
    nn.Linear(5,4),
    nn.LeakyReLU(),
    nn.Linear(4,3),
    nn.LeakyReLU(),
    nn.Linear(3, y[0].size(-1))
)
optimizer = optim.Adam(model.parameters())

n_epochs = 4000
batch_size = 256
print_interval = 100

from copy import deepcopy
lowest_loss = np.inf
best_model = None

early_stop = 100
lowest_epoch = np.inf

train_history, valid_history = [], []

for i in range(n_epochs):
    # Shuffle before mini-batch split.
    indices = torch.randperm(x[0].size(0))
    x_ = torch.index_select(x[0], dim = 0, index = indices)
    y_ = torch.index_select(y[0], dim = 0, index = indices)

    x_ = x_.split(batch_size, dim = 0)
    y_ = y_.split(batch_size, dim = 0)

    train_loss, valid_loss = 0, 0
    y_hat = []

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += float(loss)

    train_loss = train_loss / len(x_)

# You need to declare to PYTORCH to stop build the computation graph.
    with torch.no_grad():
        # You don't need to shuffle the validation set.
        # Only split is needed.
        x_ = x[1].split(batch_size, dim = 0)
        y_ = y[1].split(batch_size, dim = 0)

        valid_loss = 0

        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = F.mse_loss(y_hat_i, y_i)

            valid_loss += loss
            
            y_hat += [y_hat_i]

    valid_loss = valid_loss / len(x_)

    # Log each loss to plot after training is done.
    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i+1) % print_interval == 0:
        print('Epoch %d: train loss=%.4e valid_loss=%.4e lowest_loss=%.4e' %
              (
                  i+1,
                  train_loss,
                  valid_loss,
                  lowest_loss
              ))
        
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i

        # 'State_dict()' returns model weights as key-value.
        # Take a deep copy, if the valid loss is lowest ever.
        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break
print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))

# Load best epoch's model
model.load_state_dict(best_model)
#%% 3. 손실 곡선 확인
plot_from = 10

plt.figure(figsize = (20, 10))
plt.grid(True)
plt.title("Train / Valid Loss History")
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:],
)
plt.yscale('log')
plt.show()
# %%  4. 결과 확인
test_loss = 0
y_hat = []

with torch.no_grad():
    x_ = x[2].split(batch_size, dim = 0)
    y_ = y[2].split(batch_size, dim = 0)

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        test_loss += loss # Gradient is already detached.

        y_hat += [y_hat_i]

test_loss = test_loss / len(x_)
y_hat = torch.cat(y_hat, dim = 0)

sorted_history = sorted(zip(train_history, valid_history), key = lambda x: x[1])

print("Train loss: %.4e" % sorted_history[0][0])
print("Valid loss: %.4e" % sorted_history[0][1])
print("Test loss: %.4e" % test_loss)


df = pd.DataFrame(torch.cat([y[2], y_hat], dim = 1).detach().numpy(),
                  columns = ['y', 'y_hat'])
sns.pairplot(df, height = 5)
plt.show()
# %%
