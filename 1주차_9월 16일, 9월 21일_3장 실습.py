#%% 파이토치 텐서 생성하기
import torch
ft = torch.FloatTensor([[1,2], # 다차원 배열 값 넣어 텐서 생성
                        [3,4]])
print(ft)
lt = torch.LongTensor([[1,2], # 롱 타입
                       [3,4]])
bt = torch.ByteTensor([[1,0], # 바이트 타입
                       [0,1]])
print(lt)
print(bt)
x = torch.FloatTensor(3,2) # 크기 지정
print(x)
#%% 넘파이 호환
import numpy as np
x = np.array([[1,2],
              [3,4]])
print(x, type(x))

x1 = torch.from_numpy(x) # 넘파이 array -> 파이토치 tensor
print(x1, type(x1))

x2 = x1.numpy() # 파이토치 tensor -> 넘파이 array
print(x2, type(x2))

# 서로 다른 데이터 타입
# Long(정수), int 데이터 타입 사용해 정수 표현
# Byte(바이트), 불변(immutable)인 bytes와 가변(mutable)인 bytearray 데이터 타입이 있음
# Float(부동 소수점), float 데이터 타입 사용, 64비트(double-precision) 부동 소수점 수 나타냄
# 파이썬은 동적 타이핑(dynamic typing) 언어
# 파이썬에서는 데이터 타입 변환을 자동 처리 -> 변수 선언 시 데이터 타입 명시적으로 지정할 필요 없음.

print('정수:', ft.long()) # 정수로
print('실수:', lt.float()) # 실수로
# %% 텐서 크기 구하기
x = torch.FloatTensor([[[1,2], # 3x2x2 텐서 x 선언
                       [3,4]],
                       [[5,6],
                        [7,8]],
                       [[9,10],
                        [11,12]]])
print(x.size()) # list에 담겨서 나오는 텐서의 크기
print(x.shape)
print(x.size(1)) # 특정 차원의 크기를 알기 위해서
print(x.size(-1)) # 음수를 넣어주면 뒤에서부터 순서 읽어줌
print(x.dim()) # 차원의 개수
print(len(x.size()) == len(x.shape)) # size() 쓰나 shape 쓰나 똑같음.

# %% 3.4 실습: 기본 연산
# %% 요소별 산술 연산
a = torch.FloatTensor([[1,2],
                       [3,4]])
b = torch.FloatTensor([[2,2],
                       [3,3]])
print(a+b)
print(a-b)
print(a*b)
print(a/b) 
print(a**b)
print(a==b)
print(a!=b)
# %% 인플레이스 연산
# 앞서 수행한 연산들 결과 텐서는 메모리에 새롭게 할당됨
# 인플레이스 연산은 기존 텐서에 결과가 저장됨
print(a.mul(b))
print(a) # a 값 그대로
# 인플레이스 연산들은 함수명 뒤에 밑줄(underscore)_이 붙어있는 특징
print(a.mul_(b))
print(a) # 곱셈 결과가 a에 저장됨
# %% 차원 축소 연산: 합과 평균
x = torch.FloatTensor([[1,2],
                       [3,4]])
print(x.sum())
print(x.mean())
# 행렬 요소 전체의 합/ 평균-> 텐서나 행렬 아닌 스칼라 scalar 값으로 저장되므로 차원 축소로 볼 수 있음.
print(x.sum(dim=0)) # 차원을 넣어줌. dim 인자는 없어지는 차원
# dim = 0 -> 첫번째 차원을 이야기하는 것이므로 행렬의 세로축에 대해 합 연산을 수행하는 것'
# dim = -1 -> 뒤에서 첫번째 차원
#%% 브로드캐스트 연산: 크기가 다른 텐서끼리 산술 연산
print('텐서 + 스칼라')
x = torch.FloatTensor([[1,2],
                       [3,4]])
y = 1
z = x + y
print(x)
print(y)
print(z)
print(z.size())

print('\n')
print('텐서 + 벡터')
x = torch.FloatTensor([[1,2],
                       [4,8]])
y = torch.FloatTensor([3,5])
print(x, x.size())
print(y, y.size())
z = x + y
print(z)
print(z.size())

print('\n')
print('텐서 + 텐서')
x = torch.FloatTensor([[[1,2]]])
y = torch.FloatTensor([3,
                       5])
print(x, x.size())
print(y, y.size())
z = x + y
print(z, z.size())

x = torch.FloatTensor([[[1,2]]])
y = torch.FloatTensor([[3],
                       [5]])
print(x, x.size())
print(y, y.size())
z = x + y
print(z, z.size())
# %% 3.5 실습: 텐서 형태 변환
#%% View 함수
x = torch.FloatTensor([[[1,2],
                        [3,4]],
                        [[5,6],
                        [7,8]],
                        [[9,10],
                         [11,12]]])
print(x.size())
print(x.view(12))
print(x.view(3,4))
print(x.view(3,-1)) # -1 넣으면 다른 차원 값들 곱하고 남은 필요한 값 자동으로 채워짐.
y = x.view(3,4) # view함수 결과 텐서 주소 바뀌지 않음
x.storage().data_ptr() == y.storage().data_ptr()
print(x.reshape(3,4)) # view함수와 contiguous 함수를 순차적으로 호출한 것과 같음/ 텐서 주소가 다를 수 있음.
#%% Squeeze 함수
x = torch.FloatTensor([[[1,2],
                       [3,4]]])
print(x.size())
print(x.squeeze()) # 함수의 차원 크기가 1인 차원 없어줌
print(x.squeeze().size())
print(x.squeeze(0).size()) # 차원 지정. 해당 차원 크기 1이므로 삭제됨
print(x.squeeze(1).size()) # 차원 지정. 해당 차원 크기 1이 아니므로 같은 텐서 반환
#%% Unsqueeze 함수
x = torch.FloatTensor([[1,2],
                       [3,4]])
print(x.size())
print(x.unsqueeze(1).size()) # 지정된 차원의 인덱스에 차원 크기 1인 차원 삽입
print(x.unsqueeze(-1).size())
print(x.unsqueeze(2).size())
print(x.reshape(2,2,-1).size()) # reshape 사용해도 똑같이 구현 가능
#%% 9월 23일 토요일
# %% 3.6 실습: 텐서 자르기 & 붙이기
#%% 인덱싱과 슬라이싱
x = torch.FloatTensor([[[1,2],
                        [3,4]],
                       [[5,6],
                        [7,8]],
                       [[9,10],
                        [11,12]]])
print(x.size()) #[3,2,2]
print(x[0]) # 첫 번째 차원
print(x[-1])
print(x[:0]) # 두 번째 차원
print(x[1:2, 1:, :].size()) # 첫 번째 차원은 1이상 2이전, 두번째 차원 1이상부터, 마지막 차원 전부
#%% Split 함수
# 특정 차원에 대해 원하는 "크기"로 잘라줌
x = torch.FloatTensor(10,4) 
splits = x.split(4, dim=0) # 첫 번째 차원(dim = 0)) 4가 되도록 등분 10 -> 4,4,2로 쪼갬
for s in splits: # 각 등분된 텐서 크기 출력
    print(s.size()) 
#%% Chunk 함수
#  크기 상관 없이 원하는 "개수"로 잘라줌
x = torch.FloatTensor(8,4)
chunks = x.chunk(3, dim=0) # 첫번째 차원 최대한 같은 크기로 3등분 8 -> 3,3,2로 쪼갬
for c in chunks: # 각 등분된 텐서 크기 출력
    print(c.size())
#%% Index Select 함수
# 특정 차원에서 원하는 인덱스 값만 취함
x = torch.FloatTensor([[[1,1],
                        [2,2]],
                       [[3,3],
                        [4,4]],
                       [[5,5],
                        [6,6]]])
indice = torch.LongTensor([2,1])
print(x.size())
y = x.index_select(dim=0, index = indice) # 첫 번째 차원에서 2번 인덱스, 1번 인덱스 뽑기(2,1 순서로 출력됨)
print(y)
print(y.size()) # 2,2,2
#%% Concatenate 함수
# 여러 함수를 합쳐서 하나의 텐서로 만듦.
# Concatenate를 줄여서 cat 함수라고 함.
# 텐서를 합치기 위해서 다른 차원의 크기들이 같아야 함.
x = torch.FloatTensor([[1,2,3],
                       [4,5,6],
                       [7,8,9]])
y = torch.FloatTensor([[10,11,12],
                       [13,14,15],
                       [16,17,18]])
print(x.size(), y.size()) # [3,3], [3,3]

z = torch.cat([x,y], dim = 0) # 첫 번째 차원에 대해서 합치기(세로축으로 이어짐)
print(z)
print(z.size())

z = torch.cat([x,y], dim = -1) # 두 번째 차원에 대해서 합치기(가로축으로 이어짐)
print(z)
print(z.size())
#%% Stack 함수
# cat 함수와 비슷한 역할 수행
# 이어붙이기 작업이 아닌 "쌓기 작업" 수행 -> 새로운 차원을 만들어 cat 수행한 것과 같음.
# 교재 126p, 127p 그림 참고
z = torch.stack([x,y])
print(z)
print(z.size())
z = torch.stack([x,y], dim = -1) # 새롭게 생겨날 차원의 인덱스 지정해줌
print(z)
print(z.size())

# stack함수: 새로운 차원 만들기 + cat 함수 수행
# unsqueeze 함수와 cat 함수 사용해 stack 함수 구현하기
d = 0
# z = torch.stack([x,y], dim = d)
z = torch.cat([x.unsqueeze(d), y.unsqueeze(d)], dim = d)
print(z)
print(z.size())
#%% 유용한 팁
result = []
for i in range(5):
    x = torch.FloatTensor(2,2)
    result += [x]
print(result)
result = torch.stack(result)
print(result)
#%% 3.7 유용한 함수들
#%% Expand 함수
x = torch.FloatTensor([[[1,2]],
                       [[3,4]]])
print(x.size())
y = x.expand(2,3,2)
print(y)
print(y.size())
y = torch.cat([x]*3, dim = 1)
#%% Random Permutation 함수
x = torch.randperm(10)
print(x)
print(x.size())
#%% Argument Max 함수
x = torch.randperm(3**3).reshape(3,3,-1)
print(x)
print(x.size())
y = x.argmax(dim = -1)
print(y)
print(y.size())
# 마지막 차원에서 한 리스트마다 가장 큰 값의 인덱스 반환
#%% Top-k 함수
values, indices = torch.topk(x, k = 1, dim = -1)
print(values.size())
print(indices.size())
_, indices = torch.topk(x, k = 2, dim = -1)
print(indices.size())
#%% Sort 함수
_, indices = torch.topk(x, k = 2, dim = -1)
print(indices.size())
torch.Size([3,3,2])
target_dim = -1
values.indices = torch.topk(x, k = x.size(target_dim), largest = True)
print(values)
#%% Masked Fill 함수
x = torch.FloatTensor([i for i in range(3**2)]).reshape(3, -1)
print(x)
print(x.size())

mask = x > 4
print(mask)
y = x.masked_fill(mask, value = -1)
print(y)
#%% Ones & Zeros 함수
print(torch.ones(2,3))
print(torch.zeros(2,3))

x = torch.FloatTensor([[1,2,3],
                       [4,5,6]])
print(x.size())
print(torch.ones_like(x))
print(torch.zeros_like(x))
# %%
