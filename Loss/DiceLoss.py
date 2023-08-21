import torch.nn as nn
import torch.nn.functional as F
import torch

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)  #确定reshape成num行 但-1代表不确定列
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth) #0按列求和 1按行求和
        score = 1 - score.sum() / num
        return score