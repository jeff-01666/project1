import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Active_Contour_Loss(nn.Module):
    def __init__(self):
        super(Active_Contour_Loss,self).__init__()
    
    def forward(self,y_pred,y_true):
        epsilon = 1e-8
        w=1
        lambdaP = 1

        x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]
        y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

        delta_x = x[:,:,1:,:-2]**2
        delta_y = y[:,:,:-2,1:]**2
        delta_u = torch.abs(delta_x+delta_y)

        lenth = w * torch.sum(torch.sqrt(delta_u + epsilon))

        C_1 = torch.ones((512,512)).cuda()
        C_2 = torch.zeros((512,512)).cuda()

        region_in = torch.abs(torch.sum(y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2)))
        region_out = torch.abs(torch.sum((1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2)))

        return lenth + lambdaP * (region_in + region_out)

        


