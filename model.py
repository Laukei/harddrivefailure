import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11,10)
        self.fc2 = nn.Linear(10,2)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
        return X

    def predict(self, X):
        pred = F.softmax(self.forward(X))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)