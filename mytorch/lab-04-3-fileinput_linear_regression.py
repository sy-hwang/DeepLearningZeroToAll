import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

data = np.loadtxt('data-01-test-score.csv', delimiter=',')
x_train = torch.tensor(data[:, :3], dtype=torch.float32)
y_train = torch.tensor(data[:, [-1]], dtype=torch.float32)

#model 생성
model = nn.Linear(3, 1, bias=True)

#손실함수, 최적화함수
criterian = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=5*(1e-5))

#학습
epoch = 5000
for step in range(epoch):
    model.zero_grad()
    
    hypothesis = model(x_train)
    cost = criterian(hypothesis, y_train)
    cost.backward()
    optimizer.step()

    if step%20 == 0:
        print(f"[{step}] cost:{cost.data.numpy()}")


# Ask my score
print("Your score will be ", model((torch.tensor([[53,46,55]], dtype=torch.float32))).data.numpy())
print("Other scores will be ", model((torch.tensor([[79,80,73], [96,93,95]], dtype=torch.float32))).data.numpy())