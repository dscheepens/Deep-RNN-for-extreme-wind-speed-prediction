#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decoder.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   decoder
'''

from torch import nn
from utils import make_layers
import torch


class Decoder(nn.Module):
    def __init__(self, subnets, rnns, num_layers):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self.num_layers = num_layers

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=12)
        #print('decoder1:\t',inputs.shape)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        #print('decoder2:\t',inputs.shape)
        inputs = subnet(inputs)
        #print('decoder3:\t',inputs.shape)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        #print('decoder4:\t',inputs.shape)
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage%s'%self.num_layers),
                                       getattr(self, 'rnn%s'%self.num_layers))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,4,32,32
        inputs = inputs.transpose(1, 2)  # to B,4,S,32,32
        return inputs

