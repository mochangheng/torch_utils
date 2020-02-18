import torch
import torch.nn as nn

def build_fc(in_features, out_features, activation='relu'):
    fc = nn.Linear(in_features, out_features)
    if activation == 'relu':
        activ = nn.ReLU()
    else:
        raise NotImplementedError("Not implemented.")
    return nn.Sequential(fc, activ)