import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#load data

dataset = np.loadtxt('filename', delimeter = ',')

#input values
x = dataset[:, 0:8]
#output values
y = dataset[:, 8]

x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32).reshape(-1,1)

#create model MOFDIFY LAYERS AS NEEDED

model = nn.Sequential(nn.Linear(8, 12),
                      nn.ReLU(),
                      nn.Linear(12, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1),
                      nn.Sigmoid())

#create loss and optimization functions
loss_fn = nn.CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#establish target batches and epocs
n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(x), batch_size):
        Xbatch = x[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(x)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")




