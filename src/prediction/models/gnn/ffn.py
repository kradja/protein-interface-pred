from torch.nn import Linear
import torch.nn.functional as F
import torch


class FFN_2L(torch.nn.Module):
    def __init__(self, n_input_features, h, n_classes):
        super().__init__()
        self.linear_l1 = Linear(n_input_features, h)
        self.linear_l2 = Linear(h, n_classes)

    def forward(self, x):
        x = self.linear_l1(x)
        x = F.relu(x)
        x = self.linear_l2(x)
        return x
        # return F.log_softmax(x)