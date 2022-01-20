# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *


# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.conv_layer3 = self._conv_layer_set(32, 64)
        self.conv_layer4 = self._conv_layer_set(64, 128)
        
        self.deconv_layer1 = self._deconv_layer_set(128, 64)
        self.deconv_layer2 = self._deconv_layer_set(64, 32)
        self.deconv_layer3 = self._deconv_layer_set(32, 16)
        self.deconv_layer4 = self._deconv_layer_set(16, 1)
       
        self.dropout = nn.Dropout(p=0.0)    
    
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2)), # h,w decrease (3,5,5)
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.1),
            
            
        )
        return conv_layer
    
    def _deconv_layer_set(self, in_c, out_c):
        deconv_layer = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=(3,5,5), padding=(1,2,2), output_padding = (0,1,1), stride=(1,2,2)),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.1),
            
            
        )
        return deconv_layer
        
    def encode(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        return x
        
    def decode(self, x):
        x = self.deconv_layer1(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer3(x)
        x = self.deconv_layer4(x)        
        return x 
    
    def forward(self, x):
        x = self.encode(x)
        #x = self.dropout(x)
        x = self.decode(x)
        x = x.squeeze()
        return x
    
def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))
