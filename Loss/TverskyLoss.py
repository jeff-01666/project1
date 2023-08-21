import torch.nn as nn
import torch.nn.functional as F
import torch

class Tversky_Loss(nn.Module):

    '''
    T(A,B) = |A^B| / (|A^B| + a|A-B| +b|B-A|)

    a + b = 1

    |A^B| = TP

    |A-B| = FP

    |B-A| = FN

    a = 0.3 
    '''
    def __init__(self, alpha=0.7, size_average=True):
        super(Tversky_Loss, self).__init__()
        self.alpha = alpha
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)  #确定reshape成num行 但-1代表不确定列
        m2 = targets.view(num, -1)
        TP = (m1 * m2).sum(1)
        FN = (m2 * (1-m1)).sum(1)
        FP = ((1-m2) * m1).sum(1)

        return 1 - ((TP + smooth) / (TP + self.alpha * FN + (1-self.alpha) * FP + smooth)).sum() / num
