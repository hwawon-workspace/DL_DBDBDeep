# 13장 실습
# 13.3 Deep Binary Classification
#%% 1. 데이터 준비
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['class'] = cancer.target # 정답값 class 열에 저장

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터셋 분할
data = torch.from_numpy(df.values).float()
x = data[:, :-1]
y = data[:, -1:]

# Train / Valid / Test ratio 6:2:2
ratios = [.6, .2, .2]
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]
print("Train %d / Valid %d / Test %d samples." %(train_cnt, valid_cnt, test_cnt))

# 랜덤하게 섞어서 데이터 나누기
indices = torch.randperm(data.size(0))
x = torch.index_select(x, dim = 0, index = indices)
y = torch.index_select(y, dim = 0, index = indices)
x = x.split(cnts, dim = 0)
y = y.split(cnts, dim = 0)
for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())

# 학습 데이터 기준 표준 스케일링 학습 -> 학습/검증/테스트 데이터셋에 똑같이 적용
scaler = StandardScaler()
scaler.fit(x[0].numpy())
x = [torch.from_numpy(scaler.transform(x[0].numpy())).float(),
torch.from_numpy(scaler.transform(x[1].numpy())).float(),
torch.from_numpy(scaler.transform(x[2].numpy())).float()]
# %% 2. 학습 코드 구현
# nn.Sequential 클래스 활용해 모델 구현
# 선형계층, 리키렐루 차례대로 이어주기 -> 모델 구조 마지막에 시그모이드 넣기
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 25),
    nn.LeakyReLU(),
    nn.Linear(25, 20),
    nn.LeakyReLU(),
    nn.Linear(20, 15),
    nn.LeakyReLU(),
    nn.Linear(15, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 5),
    nn.LeakyReLU(),
    nn.Linear(5, y[0].size(-1)),
    nn.Sigmoid()) # 마지막은 시그모이드!
optimizer = optim.Adam(model.parameters()) # 아담 옵티마이저에 선언한 모델 가중치 파라미터 등록

# 하이퍼파라미터 설정
n_epochs = 10000 # 조기 종료 걸어두었으니 크게 잡아봄
batch_size = 32
print_interval = 10
early_stop = 100
lowest_loss = np.inf
best_model = None
lowest_epoch = np.inf

# 모델 학습 이터레이션진행 반복문 코드
train_history, valid_history = [], []
for i in range(n_epochs):
    indices = torch.randperm(x[0].size(0))
    x_ = torch.index_select(x[0], dim = 0, index = indices)
    y_ = torch.index_select(y[0], dim = 0, index = indices)
    x_ = x_.split(batch_size, dim = 0)
    y_ = y_.split(batch_size, dim = 0)
    train_loss, valid_loss = 0, 0
    y_hat = []
    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i) # 손실함수 BCE 사용
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss)
    train_loss = train_loss / len(x_)
    with torch.no_grad():
        x_ = x[1].split(batch_size, dim = 0)
        y_ = y[1].split(batch_size, dim = 0)
        valid_loss = 0
        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = F.binary_cross_entropy(y_hat_i, y_i)
            valid_loss += float(loss)
            y_hat += [y_hat_i]
    valid_loss = valid_loss / len(x_)
    train_history += [train_loss]
    valid_history += [valid_loss]
    if (i+1) % print_interval == 0:
        print('Epoch %d: train_loss = %.4e valid_loss = %.4e lowest_loss = %.4e' %(i+1, train_loss, valid_loss, lowest_loss))
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i
        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i+1:
            print("There is no improvement during last %d epochs." % early_stop)
            break
print("The best validation loss from epoch %d: %.4e" %(lowest_epoch + 1, lowest_loss))
model.load_state_dict(best_model)
# %% 3. 손실 곡선 확인
plot_from = 2
plt.figure(figsize = (20,10))
plt.grid(True)
plt.title("Train / Valid Loss History")
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:]
)
plt.yscale('log')
plt.show()
# %% 4. 결과 확인
# 테스트 데이터셋에 대해 평균 손실 값 구해보기
test_loss = 0
y_hat = []
with torch.no_grad():
    x_ = x[2].split(batch_size, dim = 0)
    y_ = y[2].split(batch_size, dim = 0)
    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i ,y_i)
        test_loss += loss
        y_hat += [y_hat_i]
test_loss = test_loss / len(x_)
y_hat = torch.cat(y_hat, dim = 0)
sorted_history = sorted(zip(train_history, valid_history), key = lambda x: x[1])
print("Train loss: %.4e" % sorted_history[0][0])
print("Valid loss: %.4e" % sorted_history[0][1])
print("Test loss: %.4e" % test_loss)

# 분류 정확도 계산. 임계값 0.5 가정
correct_cnt = (y[2] == (y_hat > .5)).sum()
total_cnt = float(y[2].size(0))
print('Test Accuracy: %.4f' %(correct_cnt / total_cnt))

# 예측 값 분포도 히스토그램
df = pd.DataFrame(torch.cat([y[2], y_hat], dim = 1).detach().numpy(), columns = ['y', 'y_hat'])
sns.histplot(df, x = 'y_hat', hue = 'y', bins = 50, stat = 'probability')
plt.show()

# AUROC 계산
from sklearn.metrics import roc_auc_score
roc_auc_score(df.values[:, 0], df.values[:, 1])
# %% 
# 13.7 Deep Classification # 다중 클래스 분류
#%% 1. 데이터 준비
# MNIST 데이터셋 분류
# 0 ~ 9 손글씨 숫자
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 파이토치 비전에서 제공하는 datasets 패키지에서 MNIST 데이터셋 부르기
train = datasets.MNIST('../data', train = True, download = True,
transform = transforms.Compose([
    transforms.ToTensor()
]))
test = datasets.MNIST(
    '../data', train = False,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

# 데이터 샘플 시각화 가능한 함수
def plot(x):
    img = (np.array(x.detach().cpu(), dtype = 'float')).reshape(28, 28)
    plt.imshow(img, cmap = 'gray')
    plt.show()
plot(train.data[0]) # 학습 데이터셋의 첫번째 샘플 집어넣어보기

# 0 ~ 255까지 숫자로 이뤄진 그레이스케일(gray scale)로 구성됨, 28*28 크기의 픽셀들로 이뤄짐
# 255로 나눠주면 0 ~ 1 값으로 정규화 가능
x = train.data.float() / 255.
y = train.targets
x = x.view(x.size(0), -1)
print(x.shape, y.shape)

# 하드코딩(hard_coding)(소스코드에 직접 특정 상숫값 적어 넣는 것) 최소화
input_size = x.size(-1)
output_size = int(max(y)) + 1
print('input_size: %d, output_size: %d' %(input_size, output_size))

# Train / Valid ratio
ratios = [.8, .2]
train_cnt = int(x.size(0) * ratios[0])
valid_cnt = int(x.size(0) * ratios[1])
test_cnt = len(test.data)
cnts = [train_cnt, valid_cnt]
print('Train %d / Valid %d / Test %d samples.' %(train_cnt, valid_cnt, test_cnt))
indices = torch.randperm(x.size(0))
x = torch.index_select(x, dim = 0, index = indices)
y = torch.index_select(y, dim = 0, index = indices)
x = list(x.split(cnts, dim = 0))
y = list(y.split(cnts, dim = 0))
x += [(test.data.float() / 255.).view(test_cnt, -1)]
y += [test.targets]
for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())
# %% 2. 학습 코드 구현
model = nn.Sequential(
    nn.Linear(input_size, 500), # input_size : 784
    nn.LeakyReLU(),
    nn.Linear(500, 400),
    nn.LeakyReLU(),
    nn.Linear(400, 300),
    nn.LeakyReLU(),
    nn.Linear(300, 200),
    nn.LeakyReLU(),
    nn.Linear(200, 100),
    nn.LeakyReLU(),
    nn.Linear(100, 50),
    nn.LeakyReLU(),
    nn.LeakyReLU(),
    nn.Linear(50, output_size), # output_size : 10(클래스 개수)
    nn.LogSoftmax(dim = -1)) # NLL 손실 함수 사용 위해 모델 마지막에 로그소프트맥스 함수 사용

optimizer = optim.Adam(model.parameters())
crit = nn.NLLLoss()

# 이제까지는 작은 데이터셋, 작은 모델 위주 실습했기 때문에 CPU에서 학습 가능했음
# 많은 양의 데이터, 큰 모델 학습할 것이므로 GPU에서 학습 진행.
# CUDA 활용 가능할 때 GPU를 기본 디바이스ㅂ로 지정하는 코드
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

model = model.to(device)
x = [x_i.to(device) for x_i in x]
y = [y_i.to(device) for y_i in y]

# 학습에 필요한 하이퍼파라미터, 변수 초기화
n_epochs = 1000
batch_size = 256
print_interval = 10
lowest_loss = np.inf
best_model = None
early_stop = 50
lowest_epoch = np.inf

train_history, valid_history = [], []
for i in range(n_epochs):
    indices = torch.randperm(x[0].size(0)).to(device)
    x_ = torch.index_select(x[0], dim = 0, index = indices)
    y_ = torch.index_select(y[0], dim = 0, index = indices)
    x_ = x_.split(batch_size, dim = 0)
    y_ = y_.split(batch_size, dim = 0)
    train_loss, valid_loss = 0, 0
    y_hat = []
    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = crit(y_hat_i, y_i.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss)
    train_loss = train_loss / len(x_)
    with torch.no_grad():
        x_ = x[1].split(batch_size, dim = 0)
        y_ = y[1].split(batch_size, dim = 0)
        valid_loss = 0
        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = crit(y_hat_i, y_i.squeeze())
            valid_loss += float(loss)
            y_hat += [y_hat_i]
    valid_loss = valid_loss / len(x_)
    train_history += [train_loss]
    valid_history += [valid_loss]
    if (i+1) % print_interval == 0:
        print('Epoch %d: train_loss = %.4e valid_loss = %.4e lowest_loss = %.4e' %(i+1, train_loss, valid_loss, lowest_loss))
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i 
        best_model = deepcopy(model.state_dict())
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i+1:
            print('There is no improvement during last %d epochs.' %early_stop)
            break
print('The best validation loss from epoch %d: %.4e' %(lowest_epoch + 1, lowest_loss))
model.load_state_dict(best_model)
# %% 3. 손실 곡선 확인 
plot_from = 0
plt.figure(figsize = (20,20))
plt.grid(True)
plt.title('Train / Valid Loss History')
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:]
)
plt.yscale('log')
plt.show()
# %% 4. 결과 확인
test_loss = 0
y_hat = []
with torch.no_grad():
    x_ = x[-1].split(batch_size, dim = 0)
    y_ = y[-1].split(batch_size, dim = 0)
    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = crit(y_hat_i, y_i.squeeze())
        test_loss += loss
        y_hat += [y_hat_i]
test_loss = test_loss / len(x_)
y_hat = torch.cat(y_hat, dim = 0)
print('Test loss: %.4e' %test_loss)

# 분류 문제 다루고 있으므로 신경망 마지막 계층 확률값 또는 로그 확률 값 나타냄
# 마지막 계층 출력값 중 가장 높은 값 가진 클래스 인덱스: 모델이 예측한 클래스 인덱스 -> argmax 함수 통해 구현
correct_cnt = (y[-1].squeeze() == torch.argmax(y_hat, dim = -1)).sum()
total_cnt = float(y[-1].size(0))
print('Test Accuracy: %.4f' %(correct_cnt / total_cnt))

import pandas as pd
from sklearn.metrics import confusion_matrix

# 혼동 행렬 출력
pd.DataFrame(confusion_matrix(y[-1], torch.argmax(y_hat, dim = -1)),
index = ['true_%d' % i for i in range(10)],
columns = ['pred_%d' % i for i in range(10)])