from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell

# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]

def convlstm_encoder_params(num_layers, input_len):

    if num_layers==2: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),              
            ],
            [
                CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
            ]
        ]
    if num_layers==3: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
            ],
            [
                CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
            ]
        ]
    if num_layers==4: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
                OrderedDict({'conv4_leaky_1': [128, 128, 3, 2, 1]}),
            ],
            [
                CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
                CLSTM_cell(shape=(8,8), input_channels=128, filter_size=5, num_features=256, seq_len=input_len),
            ]
        ]
    if num_layers==5:
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
                OrderedDict({'conv4_leaky_1': [128, 128, 3, 2, 1]}),
                OrderedDict({'conv5_leaky_1': [256, 256, 3, 2, 1]}),               
            ],
            [
                CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
                CLSTM_cell(shape=(8,8), input_channels=128, filter_size=5, num_features=256, seq_len=input_len),
                CLSTM_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=input_len),
            ]
        ]
    return encoder_params
 
def convlstm_decoder_params(num_layers, output_len):
    
    if num_layers==2:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv2_leaky_1': [32, 16, 3, 1, 1],
                    'conv3_leaky_1': [16, 1, 1, 1, 0]
                }),   
            ],
            [
                CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==3:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv3_leaky_1': [32, 16, 3, 1, 1],
                    'conv4_leaky_1': [16, 1, 1, 1, 0]
                }),     
            ],
            [
                CLSTM_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==4:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv3_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv4_leaky_1': [32, 16, 3, 1, 1],
                    'conv5_leaky_1': [16, 1, 1, 1, 0]
                }),
            ],
            [
                CLSTM_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CLSTM_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==5:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv3_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv4_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv5_leaky_1': [32, 16, 3, 1, 1],
                    'conv6_leaky_1': [16, 1, 1, 1, 0]
                }),
            ],
            [
                CLSTM_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CLSTM_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CLSTM_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    return decoder_params



def convgru_encoder_params(num_layers, input_len):

    if num_layers==2: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),              
            ],
            [
                CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CGRU_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
            ]
        ]
    if num_layers==3: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
            ],
            [
                CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CGRU_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CGRU_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
            ]
        ]
    if num_layers==4: 
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
                OrderedDict({'conv4_leaky_1': [128, 128, 3, 2, 1]}),
            ],
            [
                CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CGRU_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CGRU_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
                CGRU_cell(shape=(8,8), input_channels=128, filter_size=5, num_features=256, seq_len=input_len),
            ]
        ]
    if num_layers==5:
        encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
                OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
                OrderedDict({'conv4_leaky_1': [128, 128, 3, 2, 1]}),
                OrderedDict({'conv5_leaky_1': [256 , 256, 3, 2, 1]}),               
            ],
            [
                CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=32, seq_len=input_len),
                CGRU_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64, seq_len=input_len),
                CGRU_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=128, seq_len=input_len),
                CGRU_cell(shape=(8,8), input_channels=128, filter_size=5, num_features=256, seq_len=input_len),
                CGRU_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=input_len),
            ]
        ]
    return encoder_params
 
def convgru_decoder_params(num_layers, output_len):
    
    if num_layers==2:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv2_leaky_1': [32, 16, 3, 1, 1],
                    'conv3_leaky_1': [16, 1, 1, 1, 0]
                }),   
            ],
            [
                CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CGRU_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==3:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv3_leaky_1': [32, 16, 3, 1, 1],
                    'conv4_leaky_1': [16, 1, 1, 1, 0]
                }),     
            ],
            [
                CGRU_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CGRU_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==4:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv3_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv4_leaky_1': [32, 16, 3, 1, 1],
                    'conv5_leaky_1': [16, 1, 1, 1, 0]
                }),
            ],
            [
                CGRU_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CGRU_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CGRU_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    if num_layers==5:
        decoder_params = [
            [
                OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [256, 256, 4, 2, 1]}),
                OrderedDict({'deconv3_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv4_leaky_1': [64, 64, 4, 2, 1]}),
                OrderedDict({
                    'conv5_leaky_1': [32, 16, 3, 1, 1],
                    'conv6_leaky_1': [16, 1, 1, 1, 0]
                }),
            ],
            [
                CGRU_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CGRU_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
                CGRU_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
                CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
                CGRU_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),      
            ]
        ]
    return decoder_params
