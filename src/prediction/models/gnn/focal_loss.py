import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        # convert logits to probabilities
        p = F.softmax(input, dim=-1)

        # compute cross entropy loss using F.cross_entropy with the weights
        # set reduction=None to get the loss for each sample
        ce_loss = F.cross_entropy(input=input, target=target, weight=self.alpha, reduction="none")

        # select the probabilities of the target class
        p = p.gather(index=target.unsqueeze(1), dim=1).squeeze()

        # compute focal loss
        return torch.mean((1-p)**self.gamma * ce_loss)