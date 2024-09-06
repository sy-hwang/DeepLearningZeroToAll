import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)   # for reproducibility

# 1. X and Y data
x_train = torch.tensor([[1], [2], [3]], dtype=torch.float32)
y_train = torch.tensor([[1], [2], [3]], dtype=torch.float32)

# 2. 모델 생성
# nn.Linear(1, 1)은 입력 차원과 출력 차원이 각각 1인 선형 변환을 의미합니다. 
# bias=True는 편향(bias) 항을 포함시킵니다.
model = nn.Linear(1, 1, bias=True)

# 3. 손실 함수와 최적화 함수 정의
# 손실 함수로 nn.MSELoss를 정의. MSE(Mean Squared Error, 평균 제곱 오차)
# 최적화 함수로 torch.optim.SGD를 정의. SGD(Stochastic Gradient Descent, 확률적 경사 하강법), 학습률(lr)은 0.01로 설정
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 모델 훈련
epoch = 2000
for step in range(epoch):
    model.zero_grad() #이전 단계에서 계산된 기울기를 초기화
    hypothesis = model(x_train) #x_train 데이터를 모델에 입력하여 예측 값(가설, hypothesis)을 계산
    cost = criterion(hypothesis, y_train) #예측 값(hypothesis)과 실제 값(y_train) 사이의 차이(손실)를 계산
    cost.backward() #역전파(backpropagation)를 통해 기울기(gradient)를 계산
    optimizer.step() #계산된 기울기를 바탕으로 모델의 가중치와 편향을 업데이트합니다. 이는 학습의 핵심 단계로, 모델이 손실을 최소화하도록 학습됩니다.

    #매 20번째 에포크마다 현재 학습 스텝, 손실 값, 가중치, 편향 값을 출력
    if step % 20 == 0:
        print(step, cost.data.numpy(), model.weight.data.numpy(), model.bias.data.numpy())


# Testing our model
predicted = model(torch.tensor([[5]], dtype=torch.float32))
print(predicted.data.numpy())
predicted = model(torch.tensor([[2.5]], dtype=torch.float32))
print(predicted.data.numpy())
predicted = model(torch.tensor([[1.5], [3.5]], dtype=torch.float32))
print(predicted.data.numpy())
