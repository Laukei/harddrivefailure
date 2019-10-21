import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,2)

    def forward(self, X):
        X = F.softmax(self.fc1(X))
        X = F.softmax(self.fc2(X))
        return X

    def predict(self, X):
        pred = self.forward(X)
        ans = torch.zeros(pred.shape)
        for r, c in enumerate(pred.argmax(dim=1)):
            ans[r,c] = 1
        return ans