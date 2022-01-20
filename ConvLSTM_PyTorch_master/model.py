from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers


class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        x = x.squeeze()
        #x = torch.transpose(x, 0, 1)
        #x = torch.transpose(x, 1, 2)
        return x
    