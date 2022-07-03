import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import sys
import numpy as np
import argparse
from pylab import *
from scipy.stats import percentileofscore
import gc
import pickle 

sys.path.append("ConvLSTM_PyTorch_master/")

import utils as utils
from data_loader import load_era5
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
        
        
def get_percentiles(data):
    percentiles = []
    for p in [50,75,90,95,99,99.9]:
        percentiles.append(np.percentile(data,p))
    return np.array(percentiles)

def convert_to_percentiles(chunked, data):
    result = np.zeros(chunked.shape)
    for i in range(64):
        for j in range(64):
            percentiles = get_percentiles(data[:,i,j])
            for k in range(1,7): 
                result[:,:,i,j][np.where(chunked[:,:,i,j]>=percentiles[k-1])] = k                  
    return result 

def save_scores(args, scales, thresholds, targs, preds, root, model_name, categorical=False):

    scores_scales = {'sedi':np.zeros((len(thresholds), len(scales))),
                     'bias':np.zeros((len(thresholds), len(scales))),
                     'rmse':np.sqrt(np.mean((targs-preds)**2))}
    
    scores_times = {'sedi':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'bias':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'rmse':np.array([np.sqrt(np.mean((targs[:,t]-preds[:,t])**2)) for t in range(args.frames_predict)])}

    data = np.load(os.path.join(root,'data/era5_standardised.npy'))[:24*365*args.num_years]

    print('converting to percentiles...') 
    targs = convert_to_percentiles(targs, data)
    preds = convert_to_percentiles(preds, data)
    
    for i, threshold in enumerate(thresholds):
        for j, s in enumerate(scales):
            res_scales, res_times = utils.minimum_coverage(preds, targs, args, scale=s, threshold=threshold)
            scores_scales['sedi'][i,j] = res_scales[0]
            scores_scales['bias'][i,j] = res_scales[1]
            scores_times['sedi'][i,j,:] = res_times[0,:]
            scores_times['bias'][i,j,:] = res_times[1,:]            
        
    save_dir = './saved_scores/%s'%model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir+'/scores_scales.p'), 'wb') as handle:
        pickle.dump(scores_scales, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir+'/scores_times.p'), 'wb') as handle:
        pickle.dump(scores_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 
                
def save_all_scores(root, args, persist=False):
    
    scales = [(1,1),(3,3),(5,5),(7,7),(9,9)]
    thresholds = [1, 2, 3, 4, 5, 6]
    
    model_name = str(args.loss) + '_' + str(args.num_layers) + '_' + str(args.hpa) + '_40years'
    print(model_name)

    encoder_params = convlstm_encoder_params(args.num_layers, input_len=args.frames_predict)
    decoder_params = convlstm_decoder_params(args.num_layers, output_len=args.frames_predict)
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], args.num_layers).cuda()
    net = ED(encoder, decoder)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    model_info = torch.load(os.path.join(os.path.join(root,'saved_models/%s/'%model_name), 'checkpoint.pth.tar'))
    net = ED(encoder, decoder)
    net.load_state_dict(model_info['state_dict'])
    net.eval()
    test_loader = load_era5(root=os.path.join(root,'data/'), args=args, a=None, b=None, c=args.begin_testset, training=False)
    inputs, targs, preds = utils.predict_batchwise(test_loader, net, device)
    del net, test_loader   
    
    if persist: 
        print('computing persistence scores...')
        preds = utils.get_persistence_forecast(inputs)
        del inputs
        save_scores(args, scales, thresholds, targs, preds, root, model_name='persistence')
        return
    
    save_scores(args, scales, thresholds, targs, preds, root, model_name) 
    return 


def save_all_scores_ensemble(root, args, persist=False):
    
    scales = [(1,1),(3,3),(5,5),(7,7),(9,9)]
    thresholds = [1, 2, 3, 4, 5, 6]
    
    nets = []
    
    for loss, num_layers in [['wmae',5],['wmse',5],['sera',5]]:
        args.loss = loss
        args.num_layers = num_layers
        model_name = str(args.loss) + '_' + str(args.num_layers) + '_' + str(args.hpa) + '_40years'
        
        encoder_params = convlstm_encoder_params(args.num_layers, input_len=args.frames_predict)
        decoder_params = convlstm_decoder_params(args.num_layers, output_len=args.frames_predict)
        encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
        decoder = Decoder(decoder_params[0], decoder_params[1], args.num_layers).cuda()
        net = ED(encoder, decoder)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(device)
        model_info = torch.load(os.path.join(os.path.join(root,'saved_models/%s/'%model_name), 'checkpoint.pth.tar'))
        net = ED(encoder, decoder)
        net.load_state_dict(model_info['state_dict'])
        net.eval()
        nets.append(net)
        del net  
    
    test_loader = load_era5(root=os.path.join(root,'data/'), args=args, a=None, b=None, c=args.begin_testset, training=False)    
    inputs, targs, preds = utils.predict_batchwise_ensemble(test_loader, nets, device)
    del nets, test_loader   
        
    save_scores(args, scales, thresholds, targs, preds, root, model_name='ensemble')
    
    return 


if __name__=='__main__':
    
    root = '../../../../../../mnt/data/scheepensd94dm/'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device: ", device)
        
    class args():
        def __init__(self):
            self.frames_predict=12
            self.batch_size=16
            self.hpa=1000 
            self.num_years=42
            self.begin_testset=40
            self.loss = 'wmae'
            self.num_layers = 5
    args = args()
      
    persist=False 
#     save_all_scores_ensemble(root, args, persist) # ensemble 
#     save_all_scores(root, args, persist=True)     # persistence 

# ### single models: ### 

#     for loss_name, num_layers in [['wmae',5],['wmse',5],['sera',5],['mae',4],
#                                   ['mse',5]]:
#         gc.collect()
#         args.loss = loss_name 
#         args.num_layers = num_layers
#         save_all_scores(root, args, persist)
#         persist=False

