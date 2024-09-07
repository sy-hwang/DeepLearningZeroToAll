import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = np.loadtxt('data-03-diabetes.csv', delimiter=',')
x_train = data[:, 0:-1]
y_train = data[:, [-1]]

#torch로 변환
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

#model creation
model = nn.Sequential(
    nn.Linear(8, 1, bias=True),
    nn.Sigmoid()
)
criterian = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

#학습
epoch = 10000
for step in range(epoch+1):
    model.zero_grad()
    hypothesis = model(x_train)
    cost = criterian(hypothesis, y_train)

    cost.backward()
    optimizer.step()

    if step%200 ==0 :
        print(f"[{step}] cost={cost.item()}")

predicted = (model(x_train).data > 0.5)
accuracy = (predicted==y_train).float().mean()
print("accuracy=", accuracy)