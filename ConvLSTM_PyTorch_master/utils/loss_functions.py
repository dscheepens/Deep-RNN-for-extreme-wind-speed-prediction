import torch 
import numpy as np 


def choose_loss(device, preds, targets, categories, relevances, choose=None):
    if which=='wmae':
        return weighted_loss(preds, targets, categories, loss='l1')
    elif which=='wmse':
        return weighted_loss(preds, targets, categories, loss='l2')
    elif which=='sera':
        return sera_loss(preds, targets, relevances)
    elif which=='mae':
        return standard_loss(device, loss='l1')
    elif which=='mse':
        return standard_loss(device, loss='l2')
    else: 
        raise Exception('choose must be either \'wmae\', \'wmse\', \'sera\', \'mae\', or \'mse\'!')
    

def weighted_loss(preds, targets, categories, loss='l1'):
    """
    weighted loss function
    
    preds:        Tensor holding model predictions 
    targets:      Tensor holding targets 
    categories:   Tensor holding the floored values of targets 
    loss:         'l1' for MAE loss and 'l2' for MSE loss
    """
    
    # define weights: 
    weights = [1.0, 1.0, 1.0, 1.5, 3.5, 12.6, 66.0, 418.3,418.3,418.3,418.3,418.3,418.3,418.3,418.3,418.3,418.3] 
    
    w = torch.cuda.FloatTensor(weights)[cat+3]   
    if loss=='l1':
        return (w *abs(pred - targ)).mean()
    elif loss=='l2': 
        return (w *(pred - targ)**2).mean()
    else: 
        raise Exception('loss must be either \'l1\' or \'l2\'!')

        
def sera_loss(preds, targets, relevances):
    """
    SERA loss function
    
    preds:        Tensor holding model predictions 
    targets:      Tensor holding targets 
    relevances:   Tensor holding relevance values as computed by the data loader
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

