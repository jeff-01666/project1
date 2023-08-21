import torch.nn as nn
import math

def weight_init_XavierUniform(obj):
    if isinstance(obj,nn.Conv2d) or isinstance(obj,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(obj.weight)
        if obj.bias is not None:
                obj.bias = nn.init.constant_(obj.bias, 0)
    elif isinstance(obj, nn.BatchNorm2d):
        obj.weight.data.fill_(1)
        obj.bias.data.zero_()

def weight_init_KaimingNormal(obj):
    if isinstance(obj,nn.Conv2d) or isinstance(obj,nn.ConvTranspose2d):
        nn.init.kaiming_normal_(obj.weight,nonlinearity='relu')
        if obj.bias is not None:
            obj.bias = nn.init.constant_(obj.bias, 0)

def initialize_weights(obj):
    if isinstance(obj, nn.Conv2d) or isinstance(obj,nn.ConvTranspose2d):
        n = obj.kernel_size[0] * obj.kernel_size[1] * obj.out_channels
        obj.weight.data.normal_(0, math.sqrt(2. / n))
        if obj.bias is not None:
            obj.bias.data.zero_()
    elif isinstance(obj, nn.BatchNorm2d):
        obj.weight.data.fill_(1)
        obj.bias.data.zero_()

def weight_init_HeUniform(obj):
    if isinstance(obj,nn.Conv2d) or isinstance(obj,nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(obj.weight,nonlinearity='relu')
        if obj.bias is not None:
                obj.bias = nn.init.constant_(obj.bias, 0)
    elif isinstance(obj, nn.BatchNorm2d):
        obj.weight.data.fill_(1)
        obj.bias.data.zero_()