from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell

channels = 1
classes = 4 

in1, out1 = (1, 16)
in2, out2 = (32, 32)
in3, out3 = (64, 64) 
in4, out4 = (128, 128)
in5, out5 = (256, 256)

input_len = 12
output_len = 12

# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [in1, out1, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [in2, out2, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [in3, out3, 3, 2, 1]}),
        #OrderedDict({'conv4_leaky_1': [in4, out4, 3, 2, 1]}),
        #OrderedDict({'conv5_leaky_1': [in5, out5, 3, 2, 1]}),
        
#         OrderedDict({'conv1_leaky_1': [in1, out1, 3, 1, 1]}),
#         OrderedDict({'conv2_leaky_1': [in2, out2, 3, 2, 1]}),
#         OrderedDict({'conv3_leaky_1': [in3, out3, 3, 2, 1]}),
        
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=out1, filter_size=5, num_features=in2, seq_len=input_len),
        CLSTM_cell(shape=(32,32), input_channels=out2, filter_size=5, num_features=in3, seq_len=input_len),
        CLSTM_cell(shape=(16,16), input_channels=out3, filter_size=5, num_features=in4, seq_len=input_len),
        #CLSTM_cell(shape=(8,8), input_channels=out4, filter_size=5, num_features=in5, seq_len=input_len),
        #CLSTM_cell(shape=(4,4), input_channels=out5, filter_size=5, num_features=in5, seq_len=input_len),
        
#         CLSTM_cell(shape=(16,16), input_channels=out1, filter_size=5, num_features=in2, seq_len=input_len),
#         CLSTM_cell(shape=(8,8), input_channels=out2, filter_size=5, num_features=in3, seq_len=input_len),
#         CLSTM_cell(shape=(4,4), input_channels=out3, filter_size=5, num_features=in3, seq_len=input_len),


    ]
]


 
convlstm_decoder_params = [
    [
        #OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
        #OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
        OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [32, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
        
#         OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
#         OrderedDict({
#             'conv3_leaky_1': [32, 16, 3, 1, 1],
#             'conv4_leaky_1': [16, 1, 1, 1, 0]
#         }),
        
    ],

    [
        #CLSTM_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
        #CLSTM_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
        CLSTM_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
        CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),
        
#         CLSTM_cell(shape=(4,4), input_channels=64, filter_size=5, num_features=64, seq_len=output_len),
#         CLSTM_cell(shape=(8,8), input_channels=64, filter_size=5, num_features=64, seq_len=output_len),
#         CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),
    ]
]


convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [in1, out1, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [in2, out2, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [in3, out3, 3, 2, 1]}),
        OrderedDict({'conv4_leaky_1': [in4, out4, 3, 2, 1]}),
        OrderedDict({'conv5_leaky_1': [in5, out5, 3, 2, 1]}),
        
#         OrderedDict({'conv1_leaky_1': [in1, out1, 3, 1, 1]}),
#         OrderedDict({'conv2_leaky_1': [in2, out2, 3, 2, 1]}),
#         OrderedDict({'conv3_leaky_1': [in3, out3, 3, 2, 1]}),
        
    ],

    [
        CGRU_cell(shape=(64,64), input_channels=out1, filter_size=5, num_features=in2, seq_len=input_len),
        CGRU_cell(shape=(32,32), input_channels=out2, filter_size=5, num_features=in3, seq_len=input_len),
        CGRU_cell(shape=(16,16), input_channels=out3, filter_size=5, num_features=in4, seq_len=input_len),
        CGRU_cell(shape=(8,8), input_channels=out4, filter_size=5, num_features=in5, seq_len=input_len),
        CGRU_cell(shape=(4,4), input_channels=out5, filter_size=5, num_features=in5, seq_len=input_len),
        
#         CGRU_cell(shape=(16,16), input_channels=out1, filter_size=5, num_features=in2, seq_len=input_len),
#         CGRU_cell(shape=(8,8), input_channels=out2, filter_size=5, num_features=in3, seq_len=input_len),
#         CGRU_cell(shape=(4,4), input_channels=out3, filter_size=5, num_features=in3, seq_len=input_len),


    ]
]
 
convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [256, 256, 4, 2, 1]}),
        OrderedDict({'deconv3_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({'deconv4_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'conv5_leaky_1': [32, 16, 3, 1, 1],
            'conv6_leaky_1': [16, 1, 1, 1, 0]
        }),
        
#         OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
#         OrderedDict({
#             'conv3_leaky_1': [32, 16, 3, 1, 1],
#             'conv4_leaky_1': [16, 1, 1, 1, 0]
#         }),
        
    ],

    [
        CGRU_cell(shape=(4,4), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
        CGRU_cell(shape=(8,8), input_channels=256, filter_size=5, num_features=256, seq_len=output_len),
        CGRU_cell(shape=(16,16), input_channels=256, filter_size=5, num_features=128, seq_len=output_len),
        CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=64, seq_len=output_len),
        CGRU_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),
        
#         CGRU_cell(shape=(4,4), input_channels=64, filter_size=5, num_features=64, seq_len=output_len),
#         CGRU_cell(shape=(8,8), input_channels=64, filter_size=5, num_features=64, seq_len=output_len),
#         CGRU_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=32, seq_len=output_len),
    ]
]