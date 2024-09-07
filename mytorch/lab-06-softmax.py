import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

data = np.loadtxt('data-04-zoo.csv', delimiter=',')
x_train = data[:, 0:-1]
y_train = data[:, -1]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(16, 7, bias=True),
)
criterian = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

epoch = 1000
for step in range(epoch):
    model.zero_grad()
    hypothesis = model(x_train)
    cost = criterian(hypothesis, y_train)

    cost.backward()
    optimizer.step()

    if(step%20 ==0):
        print(f"[{step}] cost={cost.item()}")

pred = torch.argmax(model(x_train), dim=1)
for p, y in zip(pred, y_train):
    print("[{}] Prediction: {} True Y: {}".format(bool(p.data.item() == y.data.item()), p.data.int().item(), y.data.int().item()))

