import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = self.alpha
        self.gamma = self.gamma


    def forward(self, input, target):
        # convert logits to probabilities
        p = F.softmax(input, dim=-1)

        # compute cross entropy loss using F.cross_entropy with the weights
        ce_loss = F.cross_entropy(input=input, target=target, weight=self.alpha)
