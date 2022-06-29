import torch 
import numpy as np 


def choose_loss(device, preds, targets, x, args):
    if args.loss=='wmae':
        return weighted_loss(preds, targets, x, loss='l1')
    elif args.loss=='wmse':
        return weighted_loss(preds, targets, x, loss='l2')
    elif args.loss=='sera':
        return sera_loss(preds, targets, x)
    elif args.loss=='mae':
        return abs(preds-targets).mean() #standard_loss(device, loss='l1')
    elif args.loss=='mse':
        return ((preds-targets)**2).mean() #standard_loss(device, loss='l2')
    else: 
        raise Exception('args.loss must be either \'wmae\', \'wmse\', \'sera\', \'mae\', or \'mse\'!')
    

def weighted_loss(preds, targets, weights, loss='l1'):
    """
    weighted loss function
    
    preds:        Tensor of size (batch_size, frames_predict, 64, 64) holding model predictions 
    targets:      Tensor of size (batch_size, frames_predict, 64, 64) holding targets 
    weights:      Tensor of size (batch_size, frames_predict, 64, 64) holding weights as computed by the data loader
    loss:         'l1' for MAE loss and 'l2' for MSE loss
    """            
    
    if loss=='l1':
        return (weights *abs(preds - targets)).mean()
    elif loss=='l2': 
        return (weights *(preds - targets)**2).mean()
    else: 
        raise Exception('loss must be either \'l1\' or \'l2\'!')

        
def sera_loss(preds, targets, relevances):
    """
    SERA loss function
    
    preds:        Tensor of size (batch_size, frames_predict, 64, 64) holding model predictions 
    targets:      Tensor of size (batch_size, frames_predict, 64, 64) holding targets 
    relevances:   Tensor of size (batch_size, frames_predict, 64, 64) holding relevance values as computed by the data loader
    """
    dt = 0.1
    sera = 0.
    for t in np.arange(0., 1.+dt, dt): 
        indices = relevances >= t 
        if indices.sum() > 0: 
            ser_t = ((preds[indices]-targets[indices])**2).mean()
            sera += ser_t*dt 
    return sera


def standard_loss(device, loss='l1'):
    """
    Standard MAE or MSE loss
    
    loss:   'l1' for MAE loss and 'l2' for MSE loss
    """
    
    if loss=='l1':
        return torch.nn.L1Loss().to(device)
    elif loss=='l2':
        return torch.nn.MSELoss().to(device)
    else: 
        raise Exception('loss must be either \'l1\' or \'l2\'!')
        
        
def encoder_decoder_loss(preds, targets, model, c1=1.0, c2=1.0):
    """
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Flexible_Spatio-Temporal_Networks_CVPR_2017_paper.pdf
    """
    #encoder loss 
    loss_encoder = ((model.encode(preds.unsqueeze(2))-model.encode(targets.unsqueeze(2)))**2).mean()
    #decoder loss
    loss_decoder = ((preds - targets)**2).mean()
    return c1*loss_encoder + c2*loss_decoder

