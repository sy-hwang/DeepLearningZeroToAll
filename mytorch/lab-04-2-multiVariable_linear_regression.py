import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)   # for reproducibility

# X and Y data
x_data = torch.tensor([[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]])
y_data = torch.tensor([[152.], [185.], [180.], [196.], [142.]])

model = nn.Linear(3, 1, bias=True)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

epoch = 5000
for step in range(epoch):
    model.zero_grad()
    hypothesis = model(x_data)
    cost = criterion(hypothesis, y_data)
    cost.backward()
    optimizer.step()
    
    if step%10 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction:\n", hypothesis.data.numpy())
