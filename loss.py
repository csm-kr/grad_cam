import torch
from torch import nn


class Multi_Classificaion_Loss(nn.Module):
    def __init__(self):
        super(Multi_Classificaion_Loss, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, x, y):
        x = torch.sigmoid(x)
        loss = self.bce(x, y)
        return loss
