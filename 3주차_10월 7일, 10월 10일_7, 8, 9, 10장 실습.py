# 7.3 선형회귀
#%% 1. 데이터 준비
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 보스턴 주택 가격 데이터셋(Boston house prices dataset)에 대한 설명 출력
# 506개 샘플, 13개 속성(attribute), 이에 대한 타깃 값
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR) 

# EDA를 위해 데이터 프레임으로 변환 후 데이터 일부 확인
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['TARGET'] = boston.target
print(df.tail())

# 페이 플롯(pair plot)을 그려 각 속성 분포, 속성 사이 선형적 관계 유무 파악
sns.pairplot(df)
plt.show()

# 선형성을 띠는 일부 속성 추려서 다시 페어 플롯 그리기
cols = ['TARGET', 'INDUS', 'RM', 'LSTAT', 'NOX', 'DIS']
sns.pairplot(df[cols])
plt.show()
# %% 2. 학습 코드 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data = torch.from_numpy(df[cols].values).float()
# Numpy 데이터 -> 파이토치 실수형 텐서 변환
print(data.shape)

# 데이터 입력 x와 출력 y로 나눠주기
y = data[:, :1] # 출력(타겟 값)
x = data[:, 1:] # 입력(나머지 값들)
print(x.shape, y.shape)

# 학습에 필요한 설정값 지정
n_epochs = 2000 # 에포크: 모델이 학습 데이터 전체를 한 번 학습하는 것
learning_rate = 1e-3
print_interval = 100

# 모델 생성
# 텐서 x 마지막 차원 크기: 선형 계층의 입력 크기
# 텐서 y 마지막 차원 크기: 선형 계층 출력 크기
model = nn.Linear(x.size(-1), y.size(-1))
print(model)
# Linear(in_features=5, out_features=1, bias=True) 이렇게 뜸

# 옵티마이저 생성(파이토치에서 제공하는 옵티마이저 클래스 통해 경사하강법 구현)
# backward 함수 호출 후 옵티마이저 객체에서 step 함수 호출 -> 경사하강 1번 구현
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# 지정해준 에포크만큼 for 반복문 이용해 최적화 수행
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if (i+1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' %(i+1, loss))
# 손실값 29로 수렴
# %% 3. 결과 확인
df = pd.DataFrame(torch.cat([y, y_hat], dim = 1).detach_().numpy(),
                  columns = ['y', 'y_hat'])
sns.pairplot(df, height = 5)
plt.show()
# %%
# 8.5 로지스틱 회귀
#%% 1. 데이터 준비
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 위스콘신 유방암 데이터셋(30개의 속성, 유방암 여부 예측해야 함)
# 10개 속성에 대한 평균, 표준편차, 최악값이 각각 나타나서 30개
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.DESCR)

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['class'] = cancer.target

sns.pairplot(df[['class'] + list(df.columns[:10])]) 
plt.show()

sns.pairplot(df[['class'] + list(df.columns[10:20])])
plt.show()

sns.pairplot(df[['class'] + list(df.columns[20:30])])
plt.show()

cols = ['mean radius', 'mean texture', 'mean smoothness', 'mean compactness', 'mean concave points', 'worst radius', 'worst texture', 'worst smoothness', 'worst compactness', 'worst concave points', 'class']

for c in cols[:-1]:
    sns.histplot(df, x = c, hue = cols[-1], bins = 50, stat = 'probability')
    plt.show()
# %% 2. 학습 코드 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()
print(data.shape)

# Split x and y.
x = data[:, :-1]
y = data[:, -1:]
print(x.shape, y.shape)

# Define configurations
n_epochs = 200000
learning_rate = 1e-2
print_interval = 10000

# Define costum model
# 이번에는 선형 계층 뿐만 아니라 시그모이드 함수도 포함되어야 함
# nn.Module 상속받은 자식 클래스 정의 시 보통 두개의 함수(메서드)를 오버라이드 함
# __init__ 함수 통해 모델 구성시 필요한 내부 모듈(ex. 선형계층) 미리 선언
# forward 함수는 계산 수행
class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim  = output_dim

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.Sigmoid() # 활성함수로 시그모이드 도입한 것

    def forward(self, x):
        y = self.act(self.linear(x))
        return y
    
model = MyModel(input_dim = x.size(-1), output_dim = y.size(-1))
crit = nn.BCELoss() # BCE 손실 함수
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

for i in range(n_epochs):
    y_hat = model(x)
    loss = crit(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i+1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' %(i+1, loss))
# %% 3. 결과 확인
# 분류 문제이므로 분류 예측 결과에 대한 정확도 평가 가능
correct_cnt = (y == (y_hat > .5)).sum() # 이게 뭐지?
total_cnt = float(y.size(0))
print('Accuracy: %.4f' % (correct_cnt / total_cnt))
df = pd.DataFrame(torch.cat([y, y_hat], dim = 1).detach().numpy(), columns = ['y', 'y_hat'])
sns.histplot(df, x = 'y_hat', hue = 'y', bins = 50, stat = 'probability')
plt.show()
# %%
#9.6 Deep Regression
#%% 1. 데이터 준비
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# 보스턴 주택 가격 데이터셋
from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['TARGET'] = boston.target

scaler = StandardScaler() # 데이터셋 각 열이 정규분포 따른다고 가정하고 표준 스케일링 적용
scaler.fit(df.values[:, :-1])
df.values[:, :-1] = scaler.transform(df.values[:, :-1]).round(4)

df.tail()
# %% 2. 학습 코드 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
y = data[:, -1:]
x = data[:, :-1]

n_epochs = 200000
learning_rate = 1e-4
print_interval = 10000

class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()
        
        # 선형 계층은 각각 다른 가중치 파라미터 가지므로 따로 선언
        self.linear1 = nn.Linear(input_dim, 3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, 3)
        self.linear4 = nn.Linear(3, output_dim)
        # 비선형 활성 함수는학습되는 파라미터를 갖지 않았으므로 한 개만 선언
        self.act = nn.ReLU() 

    def forward(self, x): # 피드포워드 연산 수행
        h = self.act(self.linear1(x))
        h = self.act(self.linear2(h))
        h = self.act(self.linear3(h))
        y = self.linear4(h) # 주의: 마지막 계층에는 활성 함수 씌우지 않기!!
        
        return y

model = MyModel(x.size(-1), y.size(-1))
print(model)

# 하나씩 계산하는 것 뿐인 단순한 모델구조를 가진 모델은 nn.Sequential 클래스 이용해 심층신경망 구현 가능
# 여기서는 LeakyReLU 재활용 대신 매번 새로운 객체를 넣어줌
model = nn.Sequential(
    nn.Linear(x.size(-1), 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3,3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1))
)
print(model)

# 선언한 모델의 가중치 파라미터들을 옵티마이저에 등록
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# 반복문 수행 -> 심층신경망을 통해 회귀 수행
# 피드포워드 및 손실 계산하고 역전파, 경사하강 수행하도록 구성되어있음
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i+1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i+1, loss))
# %% 3. 결과 확인
# 정답 텐서 y와 예측값 텐서 y_hat을 이어붙이기(concatenation)
df = pd.DataFrame(torch.cat([y, y_hat], dim = 1).detach().numpy(), columns = ['y', 'y_hat'])
sns.pairplot(df, height = 5)
plt.show()
# %%
# 10.4 SGD 적용하기
#%% 1. 데이터 준비
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# 캘리포니아 주택 가격 데이터셋
# 주택 가격 예측하기
california = fetch_california_housing()

df = pd.DataFrame(california.data, columns = california.feature_names)
df['Target'] = california.target
df.tail()

sns.pairplot(df.sample(1000))
plt.show()

scaler = StandardScaler()
scaler.fit(df.values[:, :-1])
df.values[:, :-1] = scaler.transform(df.values[:, :-1])

sns.pairplot(df.sample(1000))
plt.show()
# %% 2. 학습 코드 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
print(data.shape)

x = data[:, :-1]
y = data[:, -1:]
print(x.shape, y.shape)

n_epochs = 4000
batch_size = 256
print_interval = 200
learning_rate = 1e-2

model = nn.Sequential(
    nn.Linear(x.size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1))
)
print(model)

optimizer = optim.SGD(model.parameters(), lr = learning_rate)

for i in range(n_epochs):
    # 피드포워딩을 하기 위해 데이터셋을 랜덤하게 섞고(셔플링) 미니배치로 나누기
    # randperm 함수를 통해 새롭게 섞어줄 데이터셋의 인덱스 순서 정하기
    # index_select 함수를 통해 임의 순서로 섞인 인덱스 순서대로 데이터셋 섞기
    indices = torch.randperm(x.size(0))
    x_ = torch.index_select(x, dim = 0, index = indices)
    y_ = torch.index_select(y, dim = 0, index = indices)
    
    # split 함수를 통해 원하는 배치 사이즈로 텐서 나눠주기
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

        # loss 변수에 담긴 손실 값 텐서를 float 타입캐스팅해 단순 float 타입으로 변환한 후 train_loss 변수에 더함
        # 오토그래드 작동 원리에 의해 텐서 변수들이 모두 아직 참조 중인 상태 -> 파이썬 가비지컬렉터(garbage collector)에 의해 메모리에서 해제되지 않음
        # 메모리 누수(memory leak) 방지를 위해 중요
        # float 타입캐스팅 또는 detach 함수를 통해 AutoGrad 하기 위해 연결된 그래프 잘라내는 작업 필요
        total_loss += float(loss)

        y_hat += [y_hat_i]

    total_loss = total_loss / len(x_)
    if (i+1) % print_interval == 0:
        print('Epoch %d: loss = %4.e' %(i+1, total_loss))
        
y_hat = torch.cat(y_hat, dim = 0)
y = torch.cat(y_, dim = 0)
# %% 3. 결과 확인
df = pd.DataFrame(torch.cat([y, y_hat] , dim = 1).detach().numpy(), columns = ['y', 'y_hat'])
sns.pairplot(df, height = 5)
plt.show()
# %%
