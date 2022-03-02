#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2020/03/09 18:47:50
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   encoder
'''

from torch import nn
from utilities.utils import make_layers
import torch
import logging


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        self.num_classes = 4
        
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)
        #fc1_in = int(rnns[-1].shape[0]*rnns[-1].shape[1]*rnns[-1].num_features)
        #fc1_out = int(fc1_in/4)
        #self.fc1 = nn.Linear(fc1_in, fc1_out)
        #self.fc2 = nn.Linear(fc1_out, self.num_classes)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        #print('encoder1:\t', inputs.shape)
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        #print('encoder2:\t', inputs.shape)
        inputs = subnet(inputs)
        #print('encoder3:\t', inputs.shape)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        #print('encoder4:\t', inputs.shape)
        outputs_stage, state_stage = rnn(inputs, None)
        #print('encoder5:\t', outputs_stage.shape)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            #print('stage' + str(i), 'rnn' + str(i))
            #print('inputs shape before:\t', inputs.shape)
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
            
        # for classification: 
        #print('-1:', inputs.shape)
        #inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        #inputs = self.fc1(inputs)
        #ca = self.fc2(inputs)
        #ca = torch.transpose(ca, 0, 1) 
        #ca = torch.transpose(ca, 1, 2) # (B,Classes,S)
        return tuple(hidden_states)

