import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 9
        self.fc1 = nn.Linear(11,10)
        self.fc2 = nn.Linear(10,self.output_size)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
        return X

    def predict(self, X):
        pred = F.softmax(self.forward(X))
        ans = torch.zeros(pred.shape)
        for row in zip(pred.argmax(dim=1),pred):
            row[1][row[0]] = 1
        return torch.tensor(ans)