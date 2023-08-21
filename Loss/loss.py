import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss() #二元交叉熵
        self.jaccard_weight = jaccard_weight   #1/类别数

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15 #特别的重要
            jaccard_target = (targets == 1).float()
            outputs = torch.sigmoid(outputs)
            jaccard_output = outputs.float()

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss
