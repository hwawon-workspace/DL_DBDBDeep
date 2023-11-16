#%% 15.4 분류기 모델 구현하기
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# 블록을 서브 모듈(sub-module)로 넣기 위해 클래스로 정의
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

        def get_regularizer(use_batch_norm, size):
            # use_batch_norm이 True면 nn.BatchNorm1d 계층 넣어주고, False이면 nn.Dropout 계층 넣어줌
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        # 하나의 블록은 'nn.Linear 계층, nn.LeakyReLU 활성 함수, nn.BatchNorm1d 계층/ nn.Dropout 계층' 3개로 이뤄져 nn.Sequential에 차례로 선언되어 있음.
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        y = self.block(x)
        return y

# 최종 모델
# 선언된 블록 재활용해 아키텍처 구성
# 이후 작성할 코드에 MNIST 데이터를 784차원 벡터로 변환했을 거라 가정되어 있음!!
class ImageClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes = [500, 400, 300, 200, 100],
                 use_batch_norm = True,
                 dropout_p = .3):
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"
        last_hidden_size = input_size
        blocks = []

        # hidden_sizes를 통해 필요한 블록 개수, 블록의 입출력 크기 알 수 있음.
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim = -1)
        )
    
    def forward(self, x):
        y = self.layers(x)
        return y
# %% 15.5 데이터 로딩 구현하기
# MNIST 로딩 함수(파이토치에 MNIST 쉽게 로딩하는 코드 제공해줌)
def load_mnist(is_train = True, flatten = True):
    from torchvision import datasets, transforms
    dataset = datasets.MNIST(
        '../data', train = is_train, download = True,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )
    # x,y: 이미지 데이터, 클래스 레이블 담겨있음
    x = dataset.data.float() / 255. # 원래 각 픽셀 0~255 그레이 스케일 데이터이므로 255로 나누어 0~1 데이터로 변환
    y = dataset.targets
    # flatten = True일 때 view 함수 통해 784차원 벡터로 변환
    if flatten:
        x = x.view(x.size(0), -1)
    return x, y

# 60000자의 학습 데이터 -> 학습, 검증 데이터로 나누는 함수
def split_data(x, y, train_ratio = .8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)
    y = torch.index_select(
        y,
        dim = 0,
        index = indices
    ).split([train_cnt, valid_cnt], dim = 0)
    return x, y
# %% 15.6 트레이너 클래스 구현하기
# 클래스의 가장 바깥에서 실행될 train 함수: 에포크
def train(self, train_data, valid_data, config):
    lowest_loss = np.inf
    best_model = None

    for epoch_index in range(config.n_epochs):
        train_loss = self._train(train_data[0], train_data[1], config)
        valid_loss = self._validate(valid_data[0], valid_data[1], config)

        # 검증 손실 값에 따라 현재까지 모델 저장
        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss 
            best_model = deepcopy(self.model.state_dict()) # state_dict 함수 이용해 모델의 가중치 파라미터 값을 json 형태로 변환해 리턴 -> 값 자체 복사해 best_model에 할당

        print("Epoch(%d/%d): train_loss = %.4e valid_loss = %.4e lowest_loss = %.4e" %(
            epoch_index + 1,
            config.n_epochs,
            train_loss,
            valid_loss,
            lowest_loss
        ))

    # 가중치 파라미터 json값을 load_state_dict 이용해 self.model에 다시 로딩
    # 학습 종료 후 오버피팅되지 않은 가장 좋은 상태 모델로 복원
    self.model.load_state_dict(best_model)

# 한 이터레이션 학습 위한 for문 반복 구현
def _train(self, x, y, config):
    # train 함수 호출해 모델을 학습 모드로 전환!!
    self.model.train()

    x, y - self._batchify(x, y, config.batch_size) 
    total_loss = 0

    for i, (x_i, y_i) in enumerate(zip(x,y)):
        # 피드포워드
        y_hat_i = self.model(x_i)
        loss_i = self.crit(y_hat_i, y_i.squeeze())
        # 역전파 계산
        self.optimizer.zero_grad()
        loss_i.backward()
        # 경사하강법에 의한 파라미터 업데이트
        self.optimizer.step()

        # 현재 학습 현황 출력
        # config: 가장 바깥 train.py에서 사용자 실행 시 파라미터 입력에 따른 설정값 들어있는 객체
        if config.verbose >= 2:
            print("Train Iteration(%d/%d): loss = %.4e" %(i+1, len(x), float(loss_i)))
            total_loss += float(loss_i)
        
        return total_loss / len(x)

# _batchify: 매 에포크마다 SGD 수행 위해 셔플링, 미니배치 만드는 과정    
# 검증 과정에서는 random_split 필요 없으므로 False로 넘어올 수 있음을 유의
def _batchify(self, x, y, batch_size, random_split = True):
    if random_split:
        indices = torch.randperm(x.size(0), device = x.device)
        x = torch.index_select(x, dim = 0, index = indices)
        y = torch.index_select(y, dim = 0, index = indices)
    x = x.split(batch_size, dim = 0)
    y = y.split(batch_size, dim = 0)
    return x, y

# 검증 과정을 위한 _validate 함수
# _train과 거의 비슷하게 구현되어 있으나, 가장 바깥쪽에 torch.no_grad() 호출되어 있음 유의
def _validate(self, x, y, config):
    self.model.eval()

    with torch.no_grad():
        x, y = self._batchify(x, y, config.batch_size, random_split = False)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            if config.verbose >= .2:
                print("Valid Iteration(%d/%d): loss = %.4e" %(i+1, len(x), float(loss_i)))
            total_loss += float(loss_i)
        return total_loss / len(x)
# %% 15.7 train.py 구현하기
#%% 15.8 predict.ipynb 구현하기
