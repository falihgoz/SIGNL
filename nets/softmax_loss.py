import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLossFunction(nn.Module):
    def __init__(self, in_dim=80, num_classes=2, dataset=None):
        super(SoftmaxLossFunction, self).__init__()
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(num_classes, self.in_dim))
        nn.init.kaiming_uniform_(self.weight, a=0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.bias = nn.Parameter(torch.zeros(num_classes))
        if dataset == "CFAD":
            self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.33, 0.67]))
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))

        self.logsoftmax = nn.LogSoftmax(dim=1)  # addition
        self.relu = nn.ReLU(inplace=True)  # addition

    def forward(self, x, label=None):
        assert x.size(1) == self.in_dim, "Input feature dimension must be 80"
        x = self.relu(x)
        logits = F.linear(x, self.weight, self.bias)
        logits = self.logsoftmax(logits)

        loss = self.ce(logits, label)
        _, pred = logits.max(1)
        return loss, -logits[:, 0], pred
