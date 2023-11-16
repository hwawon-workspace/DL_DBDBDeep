# 14.6 정규화
#%% 1. 데이터 준비
# 라이브러리 불러오기
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# MNIST 데이터셋 불러오기. 데이터 없다면 자동으로 다운로드할 것.
# 60000장 학습 샘플, 10000장 테스트 샘플
train = datasets.MNIST(
    '../data', train = True, download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
)
test = datasets.MNIST(
    '../data', train = False,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)
x = train.data.float() / 255.
y = train.targets
x = x.view(x.size(0), -1) # 2차원 이미지 행렬 -> 1차원 벡터로 변환
print(x.shape, y.shape) # 학습데이터의 입, 출력 텐서 크기
input_size = x.size(-1)
output_size = int(max(y)) + 1
print('input_size: %d, output_size: %d' % (input_size, output_size))

# MNIST는 테스트 데이터셋 이미 정해져 있음
# 학습 데이터셋을 8:2 비율로 학습/검증 데이터셋으로 나눔
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
# 서브 모듈 클래스 정의 코드
# 지금까지 모델들 선형 계층, 비선형 활성 함수의 반복이었음. (하나의 층: 선형 계층, 비선형 활성 함수의 조합)
# 전체 모듈에 대한 부분 모듈/ 서브 모듈로 봤을 때 입출력 크기만 바뀐 서브 모듈이 반복되고 있었던 것
# 이번엔 하나의 서브 모듈: 선형 계층 + 비선형 활성 함수 + 정규화 계층
# -> 서브 모듈을 nn.Module을 상속받아 하나의 클래스로 정의, nn.Sequential에 정의한 클래스 객체를 넣어줌)
class Block(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm = True,
                 dropout_p = .4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size): # 생성 시 배치 정규화/ 드롭아웃 중 하나 선택받음
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size) 
        # 정규화 계층, 배치 정규화이면 앞서 사용한 선형계층의 출력 크기를, 드롭아웃이면 확률 값을 넣어줘야 함.
        )
    def forward(self, x):
        y = self.block(x)
        return y

# 위에서 만든 Block 클래스를 MyModel 클래스에 활용
class MyModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm = True,
                 dropout_p = .4):
        super().__init__()

        self.layers = nn.Sequential(
            Block(input_size, 500, use_batch_norm, dropout_p),
            Block(500, 400, use_batch_norm, dropout_p),
            Block(400, 300, use_batch_norm, dropout_p),
            Block(300, 200, use_batch_norm, dropout_p),
            Block(200, 100, use_batch_norm, dropout_p),
            Block(100, 50, use_batch_norm, dropout_p),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim = -1)
        )
    
    def forward(self, x): # 피드포워드 수행
        y = self.layers(x)
        return y

# MyModel 모델 객체 선언, 프린트    
model = MyModel(input_size,
                output_size,
                use_batch_norm = True)
print(model)

# 모델 가중치 파라미터를 아담 옵티마이저에 등록, NLL 함수 선언
optimizer = optim.Adam(model.parameters())
crit = nn.NLLLoss()

# CUDA 활용 가능한 경우 GPU가 기본 디바이스가 되도록 device 변수에 집어넣고, 모델과 텐서를 원하는 디바이스로 이동 및 복사
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
model = model.to(device)
x = [x_i.to(device) for x_i in x]
y = [y_i.to(device) for y_i in y]

# 학습에 필요한 하이퍼파라미터, 변수들 초기화
n_epochs = 1000
batch_size = 256
print_interval = 10
lowest_loss = np.inf
best_model = None
early_stop = 50
lowest_epoch = np.inf

# 학습 위한 코드
train_history, valid_history = [], []
for i in range(n_epochs):
    model.train()
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

    model.eval()
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
plt.figure(figsize = (20, 10))
plt.grid(True)
plt.title("Train / Valid Loss History")
plt.plot(range(plot_from, len(train_history)), train_history[plot_from:], label = "train_history")
plt.plot(range(plot_from, len(valid_history)), valid_history[plot_from:], label = "valid_history")
plt.yscale('log')
plt.legend()
plt.show()
# 학습 손실 곡선(파란색)은 계속해서 감소, 검증 손실 곡선(주황색)은 67 에포크부터 천천히 올라감
# 성능이 얼마나 개선되었는가? 단순히 성능 개선폭을 보는 것만으로는 X
# (60 -> 70) < (97 -> 98)
# ERR(Error Reduction Rate)을 통해 상대적인 모델의 개선 폭 측정 가능
# 최소 5번 이상 같은 실험 반복하여 평균 테스트 정확도 측정한 후 ERR 계산
#%% 4. 결과 확인
# 테스트셋에서 더 좋은 성능 나왔는가
test_loss = 0
y_hat = []
model.eval()
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
print("Test Loss: %.4e" % test_loss)
correct_cnt = (y[-1].squeeze() == torch.argmax(y_hat, dim = -1)).sum()
total_cnt = float(y[-1].size(0))
print('Test Accuracy: %.4f' % (correct_cnt / total_cnt))

import pandas as pd
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y[-1], torch.argmax(y_hat, dim = -1)),
             index = ['true_%d' %i for i in range(10)],
             columns = ['pred_%d' %i for i in range(10)])
# 정규화 도입을 통해 오버피팅 최대한 지연, 일반화 성능 향상 가능을 확인
# %%
