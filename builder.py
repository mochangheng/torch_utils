import torch
import torch.nn as nn

def build_fc(in_features, out_features, activation='relu'):
    fc = nn.Linear(in_features, out_features)
    if activation == 'relu':
        activ = nn.ReLU()
    elif activation == 'none':
        activ = nn.Identity()
    else:
        raise NotImplementedError("Not implemented.")
    return nn.Sequential(fc, activ)

def build_conv(in_channels, out_channels, kernel_size, stride=1, activation='relu', batch_norm=False, norm_first=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)
    if activation == 'relu':
        activ = nn.ReLU()
    elif activation == 'none':
        activ = nn.Identity()
    else:
        raise NotImplementedError("Not implemented.")

    if batch_norm:
        if norm_first:
            norm = nn.BatchNorm2d(in_channels)
            return nn.Sequential(norm, activ, conv)
        else:
            norm = nn.BatchNorm2d(out_channels)
            return nn.Sequential(conv, norm, activ)

    return nn.Sequential(conv, activ)

def build_pool(pool_size, kernel_size=3, mode='max'):
    if mode == 'max':
        pool = nn.MaxPool2d(kernel_size, stride=pool_size, padding=kernel_size//2)
    else:
        raise NotImplementedError("Not implemented.")
    return pool
