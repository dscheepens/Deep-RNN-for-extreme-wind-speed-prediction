#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data_loader import load_era5
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import utils 
import loss_functions
import matplotlib.pyplot as plt 
from ConvCNN import CNNModel
import time 
#torch.backends.cudnn.enabled = False

root = '../../../../../../mnt/data/scheepensd94dm/'
data_root = root + 'data/'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='convlstm', help='convlstm, convgru or convcnn')
parser.add_argument('--num_layers', default=3, help='2, 3, 4 or 5')
parser.add_argument('--num_years', default=10, help='number of years of wind speed data')
parser.add_argument('--frames_predict',default=12,type=int,help='sum of predict frames')
parser.add_argument('--device',default='cpu')
parser.add_argument('--hpa',default=1000, help='1000, 925, 850 or 775')
parser.add_argument('--y1',default=None, help='90th percentile')
parser.add_argument('--y2',default=None, help='99th percentile')
# training/optimization related:
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lam', default=1., type=float, help='lambda')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
parser.add_argument('--epochs', default=200, type=int, help='sum of epochs')

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = args.device

args.model = 'convlstm'
args.epochs = 200    

if args.model == 'convlstm':
    model_name = 'convlstm.pth'
elif args.model == 'convgru':
    model_name = 'convgru.pth'
elif args.model == 'convcnn':
    model_name = 'convcnn.pth'
else:
    raise Exception('Model must be either \'convlstm\', \'convgru\' or \'convcnn\'!')
    
random_seed = 999
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

   
def train(STAMP, args, a=0.6, b=0.8):
    '''
    main function to run the training
    
    STAMP:    Stamp of this model run 
    Validation set within [a,b] where a and b<=0.8 are percentages of the dataset. Rest is used for training. [0.8,1.0] is reserved for testing.
    '''

    save_dir = root + 'save_model/' + STAMP
    run_dir = root + 'runs/' + STAMP
    
    print(args.model, 'with', args.loss, 'loss at', args.hpa, 'hpa')
    print('device:', args.device)
    print('STAMP: ', STAMP)
    print('batch size:', args.batch_size)
    print('epochs:', args.epochs)
    
    trainLoader, validLoader, testLoader = load_era5(root=data_root, args=args, a=a, b=b, c=0.8)
    
    if args.model == 'convlstm':
        encoder_params = convlstm_encoder_params(args.num_layers, input_len=args.frames_predict)
        decoder_params = convlstm_decoder_params(args.num_layers, output_len=args.frames_predict)
    elif args.model == 'convgru':
        encoder_params = convgru_encoder_params(args.num_layers, input_len=args.frames_predict)
        decoder_params = convgru_decoder_params(args.num_layers, output_len=args.frames_predict)
     
    if args.model=='convcnn':
        net = CNNModel()
    else: 
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
        decoder = Decoder(decoder_params[0], decoder_params[1], args.num_layers).to(device)
        net = ED(encoder, decoder)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0   
        
    print('Number of parameters: %s M.\n'%(utils.count_parameters(net)/1e6))
    
    with open(save_dir + "/model_params.txt", 'wt') as f:
        print('%s with %s loss.\n'%(args.model, args.loss), file=f)
        print('Number of parameters: %s.\n'%utils.count_parameters(net), file=f)
        print(net, file=f)

            
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    #pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  factor=0.5,
    #                                                  patience=4,
    #                                                  verbose=True)
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    train_accuracies = []
    valid_accuracies = [] 
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (inputVar, targetVar, relVar) in enumerate(t):
            inputs = inputVar.to(device)  
            targets = targetVar.to(device)  
            rels = relVar.to(device)
            cats = np.floor(targetVar).long().to(device)
            
            optimizer.zero_grad()
            net.train()
            pred = net(inputs.unsqueeze(2)).squeeze()
            
            loss = loss_functions.choose_loss(device, pred, targets, cats, rels, choose=args.loss, hpa=args.hpa).requires_grad_()
                        
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (inputVar, targetVar, relVar) in enumerate(t):
                #if i == 3000:
                #    break
                inputs = inputVar.to(device)
                targets = targetVar.to(device)
                rels = relVar.to(device)
                cats = np.floor(targetVar).long().to(device)
              
                pred = net(inputs.unsqueeze(2)).squeeze()
                
                loss = loss_functions.choose_loss(device, pred, targets, cats, rels, choose=args.loss, hpa=args.hpa)
               
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {valid_loss:.3f}   ')# + 
                     #f'train_acc: {train_acc:.3f} ' +
                     #f'valid_acc: {valid_acc:.3f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        #pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    np.save(save_dir + "/avg_train_losses.npy", avg_train_losses)
    np.save(save_dir + "/avg_valid_losses.npy", avg_valid_losses)
    return 


if __name__ == "__main__":
     
    args.batch_size = 16    
    args.hpa = 1000
    args.num_layers = 3

    for loss_name in ['sera']:
        model_name = loss_name+'_%s_%s'%(args.num_layers, args.hpa)
        args.loss = loss_name
        for STAMP, a, b in [[model_name+'/cv0', 0.6, 0.8],
                            [model_name+'/cv1', 0.4, 0.6], 
                            [model_name+'/cv2', 0.2, 0.4],
                            [model_name+'/cv3', 0.0, 0.2]]:
            train(STAMP, args, a, b)

    

    
    
    
    