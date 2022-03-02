import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import sys
import numpy as np
import argparse
from pylab import *

from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from utilities import utils 
from data_loader import load_era5

class args():
    def __init__(self):
        self.frames_predict=12
        self.batch_size=8
        self.num_layers=3
        self.hpa=1000 
        
        
def get_percentiles(data):
    
    l, r = data.min(), data.max()
    r = 8
    num=500
    
    n, bins = utils.function_hist(data, l, r, num)
    
    cs = np.cumsum(n)
    
    p50 = np.round(np.where(cs>0.5*sum(n))[0][0]*(r-l)/num+l, 1)
    p75 = np.round(np.where(cs>0.75*sum(n))[0][0]*(r-l)/num+l, 1)
    p90 = np.round(np.where(cs>0.9*sum(n))[0][0]*(r-l)/num+l, 1)
    p95 = np.round(np.where(cs>0.95*sum(n))[0][0]*(r-l)/num+l, 1)
    p99 = np.round(np.where(cs>0.99*sum(n))[0][0]*(r-l)/num+l, 1)
    p999 = np.round(np.where(cs>0.999*sum(n))[0][0]*(r-l)/num+l, 1)
    
    return p50, p75, p90, p95, p99, p999 

        
def save_all_scores(args):
    
    root = '../../../../../../mnt/data/scheepensd94dm/'
    
    data = np.load(os.path.join(root, 'data/adaptor.mars.internal-Horizontal_velocity_%s.npy'%args.hpa)[:24*365*8])
    data = utils.standardize_local(data)[0]
    
    percentiles = get_percentiles(data)
    print('percentiles:',percentiles)
    
    scales = [(1,1),(3,3),(5,5),(7,7),(9,9)]
    thresholds = [percentiles[-2]]
    
    for loss_name in ['wmae','wmse','sera','mae','mse']:
        model_name = loss_name + '_' + str(args.num_layers)
        print('\nmodel:',model_name,'\t started.')

        for cv in range(0,4):   
            print('cv_%s'%cv)
            encoder_params = convlstm_encoder_params
            decoder_params = convlstm_decoder_params
            encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
            decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
            net = ED(encoder, decoder)
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
            net.to(device)
            model_info = torch.load(os.path.join(root+'save_model/%s/cv%s/'%(model_name, cv), 'checkpoint.pth.tar'))
            net = ED(encoder, decoder)
            net.load_state_dict(model_info['state_dict'])
            net.eval()
            _, _, test_loader = load_era5(root=root+'data/', args=args, a=0.6, b=0.8, c=0.8)
            _, targs, preds, _, _, _ = utils.predict_batchwise(test_loader, net, device, report=False)
            del net 

            scores_scales = np.zeros((len(thresholds), len(scales)))
            scores_times = np.zeros((len(thresholds), len(scales), args.frames_predict))

            for i, threshold in enumerate(thresholds):
                #print('threshold:',threshold)

                preds_random = utils.get_random_forecast(preds>threshold, targs>threshold)   

                for j, s in enumerate(scales):
                    #print('scale:',s)

                    res_scales, res_times = utils.minimum_coverage(preds, preds_random, targs, args, 
                                                        scale=s, threshold=threshold)
                    scores_scales[i,j] = res_scales
                    scores_times[i,j,:] = res_times

            np.save(root+'saved_scores/%s/scores_scales_%s.npy'%(model_name, cv), scores_scales)
            np.save(root+'saved_scores/%s/scores_times_%s.npy'%(model_name, cv), scores_times)

            #print('model:',model_name,cv,'\t finished.')
            del _, targs, preds, preds_random, test_loader
    return 



if __name__=='__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Use device: ", device)
        
    args = args()
    
    args.hpa = 1000 
    args.num_layers = 3
    
    save_all_scores(args)
    
