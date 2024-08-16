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
    for p in [50,75,90,95,99,99.9,100]:
        percentiles.append(np.percentile(data,p))
    return np.array(percentiles)

def get_scores(fcs, obs, t0, t1):
    counts = utils.get_counts_from_binary(fcs>=t0, obs>=t1)
    bias = utils.get_bias(counts)
    sedi = utils.get_sedi(counts)
    return counts, bias, sedi

def convert_to_percentiles(test_targs, test_preds, train_targs, train_preds):
    gc.collect()
    result_targs = np.zeros(test_targs.shape)
    result_preds = np.zeros(test_preds.shape)

    for i in range(64):
        for j in range(64): 
            #p_targs = get_percentiles(train_targs[:,:,i,j])
            #p_preds = get_percentiles(train_preds[:,:,i,j])
            percentiles = get_percentiles(train_targs[:,:,i,j])

            for k in range(1,7): 
                # convert local values to local percentiles 
                
                #result_targs[:,:,i,j][test_targs[:,:,i,j]>=p_targs[k-1]] = k 
                #result_preds[:,:,i,j][test_preds[:,:,i,j]>=p_preds[k-1]] = k
                
                result_targs[:,:,i,j][test_targs[:,:,i,j]>=percentiles[k-1]] = k 
                result_preds[:,:,i,j][test_preds[:,:,i,j]>=percentiles[k-1]] = k
                                
    return result_targs, result_preds 

def compute_rmse(targs, preds, percentiles):
    rmse = np.zeros(6)
    for k in range(1,7):
        t = targs[(targs>=percentiles[k-1]) & (targs<percentiles[k])] # all targs between percentiles k-1 and k 
        p = preds[(targs>=percentiles[k-1]) & (targs<percentiles[k])] # corresponding preds 
        rmse[k-1] = np.sqrt(np.mean((t-p)**2))
    return rmse 
    

def save_scores(root, args, scales, thresholds, targs, preds, train_targs, train_preds, model_name):
    gc.collect()
        
    print('converting to percentiles...') 
    targs_perc, preds_perc = convert_to_percentiles(targs, preds, train_targs, train_preds)
    
    scores_scales = {'sedi':np.zeros((len(thresholds), len(scales))),
                     'bias':np.zeros((len(thresholds), len(scales))),
                    'csi':np.zeros((len(thresholds), len(scales))),
                    'hss':np.zeros((len(thresholds), len(scales))),
                    'far':np.zeros((len(thresholds), len(scales))),
                    'pod':np.zeros((len(thresholds), len(scales))),
                    'rmse':np.zeros((len(thresholds))),
                    }
    
    scores_times = {'sedi':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'bias':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'csi':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'hss':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'far':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'pod':np.zeros((len(thresholds), len(scales), args.frames_predict)),
                    'rmse':np.zeros((len(thresholds), args.frames_predict)),
                   }
    
    
    percentiles = get_percentiles(targs)
    scores_scales['rmse'] = compute_rmse(targs, preds, percentiles)
    for t in range(args.frames_predict): 
        scores_times['rmse'][:,t] = compute_rmse(targs[:,t],preds[:,t],percentiles)
    
    for i, threshold in enumerate(thresholds):
        for j, s in enumerate(scales):
            res_scales, res_times = utils.minimum_coverage(preds_perc, targs_perc, args, scale=s, threshold=threshold)
            scores_scales['sedi'][i,j] = res_scales[0]
            scores_scales['bias'][i,j] = res_scales[1]
            scores_scales['csi'][i,j] = res_scales[2]
            scores_scales['hss'][i,j] = res_scales[3]
            scores_scales['far'][i,j] = res_scales[4]
            scores_scales['pod'][i,j] = res_scales[5]
            
            scores_times['sedi'][i,j,:] = res_times[0,:]
            scores_times['bias'][i,j,:] = res_times[1,:]  
            scores_times['csi'][i,j,:] = res_times[2,:]  
            scores_times['hss'][i,j,:] = res_times[3,:]  
            scores_times['far'][i,j,:] = res_times[4,:]  
            scores_times['pod'][i,j,:] = res_times[5,:]  
        
    save_dir = './saved_scores_calib/%s'%model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir,'scores_scales.p'), 'wb') as handle:
        pickle.dump(scores_scales, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir,'scores_times.p'), 'wb') as handle:
        pickle.dump(scores_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 
                
def save_all_scores(root, args, model_name, persist=False):
    gc.collect()
    scales = [(1,1),(3,3),(5,5),(7,7),(9,9)]
    thresholds = [1, 2, 3, 4, 5, 6]

    encoder_params = convlstm_encoder_params(args.num_layers, input_len=args.frames_predict)
    decoder_params = convlstm_decoder_params(args.num_layers, output_len=args.frames_predict)
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], args.num_layers).cuda()
    net = ED(encoder, decoder)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    model_info = torch.load(os.path.join(os.path.join(root,'saved_models_final/%s/'%model_name), 'checkpoint.pth.tar'))
    net = ED(encoder, decoder)
    net.load_state_dict(model_info['state_dict'])
    net.eval()  
    
    print('load train_loader...')
    args.begin_testset, args.num_years = 30, 40
    train_loader = load_era5(root=os.path.join(root,'data/'), args=args, a=None, b=None, c=args.begin_testset, training=False)
    print('getting predictions...')
    if persist: 
        print('persistence...')
        train_inputs, train_targs = utils.sample_dataloader(train_loader)
        train_preds = utils.get_persistence_forecast(train_inputs)
        del train_inputs
    else: 
        train_targs, train_preds = utils.predict_batchwise(train_loader, net, device)
        
    del train_loader
    gc.collect()
    
    print('load_test_loader...')
    args.begin_testset, args.num_years = 40, 42
    test_loader = load_era5(root=os.path.join(root,'data/'), args=args, a=None, b=None, c=args.begin_testset, training=False)
    if persist: 
        print('persistence...')
        inputs, targs = utils.sample_dataloader(test_loader)
        preds = utils.get_persistence_forecast(inputs)
        del inputs
    else:
        targs, preds = utils.predict_batchwise(test_loader, net, device)
    
    del test_loader, net 
    gc.collect()

    if persist: 
        save_scores(root, args, scales, thresholds, targs, preds, train_targs, train_preds, model_name='persistence')
    else:
        save_scores(root, args, scales, thresholds, targs, preds, train_targs, train_preds, model_name) 
    return 


# def save_all_scores_ensemble(root, args, persist=False):
#     gc.collect()
#     scales = [(1,1),(3,3),(5,5),(7,7),(9,9)]
#     thresholds = [1, 2, 3, 4, 5, 6]
    
#     nets = []
    
#     for loss, num_layers in [['wmae',5],['wmse',5],['sera',5]]:
#         args.loss = loss
#         args.num_layers = num_layers
#         model_name = str(args.loss) + '_' + str(args.num_layers) + '_' + str(args.hpa) + '_40years'
        
#         encoder_params = convlstm_encoder_params(args.num_layers, input_len=args.frames_predict)
#         decoder_params = convlstm_decoder_params(args.num_layers, output_len=args.frames_predict)
#         encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
#         decoder = Decoder(decoder_params[0], decoder_params[1], args.num_layers).cuda()
#         net = ED(encoder, decoder)
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#         net.to(device)
#         model_info = torch.load(os.path.join(os.path.join(root,'saved_models_final/%s/'%model_name), 'checkpoint.pth.tar'))
#         net = ED(encoder, decoder)
#         net.load_state_dict(model_info['state_dict'])
#         net.eval()
#         nets.append(net)
#         del net  
    
#     test_loader = load_era5(root=os.path.join(root,'data/'), args=args, a=None, b=None, c=args.begin_testset, training=False)    
#     inputs, targs, preds = utils.predict_batchwise_ensemble(test_loader, nets, device)
#     del nets, test_loader   
        
#     save_scores(args, scales, thresholds, targs, preds, root, model_name='ensemble')
    
#     return 


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
            self.num_layers=4
    args = args()
    
    
# ### single models: ###  

    persist=False 
    for loss_name, num_layers in [['wmae_i',4], ['wmse_i',4], ['wmae_l',5], ['wmse_l',4], ['sera_p90',5], ['sera_p75',5],['sera_p50',5], ['mae',5],['mse',5]]:
        gc.collect()
        args.num_layers=num_layers         
        model_name = loss_name+'_'+str(num_layers)
        print(model_name)
        
        save_all_scores(root, args, model_name, persist)

#     args.num_layers=4
#     save_all_scores(root, args, model_name='wmae_i_4', persist=True) # persistence 
       
        
        
