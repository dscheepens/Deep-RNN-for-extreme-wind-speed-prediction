import torch
from torch import nn
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np 
#import pywt 


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def standardize_local(data):
    data2 = data.copy()
    m = data2.mean(0)
    s = data2.std(0)
    zvals = [m,s]
    #for j in range(len(data)):
    data2 = (data2 - m)/s
    return data2, zvals


def function_hist(a, ini, final, num):
    bins = np.linspace(ini, final, num)
    #weightsa = np.ones_like(a)/float(len(a))
    hist, bin_edges = np.histogram(np.array(a), bins) #weights = weightsa)
    return hist, bins


def chunkify(data, args):
    num_samples = len(data)

    num_input = args.frames_predict
    num_output = args.frames_predict

    from_ = 0
    to_ = num_samples - num_input - num_output
    shift_ = 12 

    inputs = []
    targets = []
    for i in range(from_, to_, shift_):
        chunked_inputs = []
        chunked_targets = []
        
        for chunk in data[i : i+num_input]:
            chunked_inputs.append(chunk)
            
        for chunk in data[i+num_input : i+num_input+num_output]:
            chunked_targets.append(chunk)
            
        inputs.append(chunked_inputs)
        targets.append(chunked_targets)
        
    inputs = np.array(inputs, dtype=float)  
    targets = np.array(targets, dtype=float)  
    
    return inputs, targets


####################################################################
#                 Data analysis related functions                  #
####################################################################
    
    
def predict_batchwise(dataloader, model, device, report=False):
    model.eval()
    inputs = []
    predictions = []
    targets = []

    for batch in dataloader:  
        inputs.append(batch[0].squeeze().numpy())
     
        batch_data = batch[0].to(device).unsqueeze(2)
        pred = model(batch_data).squeeze().detach().cpu().numpy()
        predictions.append(pred)
        
        target = batch[1].detach().cpu().numpy()
        targets.append(target)
    
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0) 
    
    if report: 
        targets = np.array(np.floor(targets),dtype=int).reshape(-1)
        predictions = np.array(np.floor(predictions),dtype=int).reshape(-1)
        acc = accuracy_score(targets, predictions)
        bacc = balanced_accuracy_score(targets, predictions)
        report = classification_report(targets, predictions, target_names=list(map(str, list(range(-2,4)))), zero_division=0)
    else: 
        acc, bacc, report = None, None, None
        
    return inputs, targets, predictions, acc, bacc, report


def get_persistence_forecast(inputs): 
    """
    generates a persistence forecast based on the last input frame
    """
    
    persist = inputs.copy()
    for i in range(inputs.shape[0]):
        last_frame = inputs[i,-1]
        for j in range(inputs.shape[1]):
            persist[i,j] = last_frame
    return persist 


def get_random_forecast(predictions, targets):
    """
    generates a random forecast with probability s to forecast an event at each instance. 
    """
       
    counts = get_counts_from_binary(predictions, targets)
    
    s = (counts['tp']+counts['fn'])/sum(counts.values())
    r = (counts['tp']+counts['fp'])/sum(counts.values())
    #print('s:',s)
    #print('r:',r)

    return np.random.choice([0, 1], size=predictions.shape, p=[1-r, r])
    
    
def get_sedi(counts): 
    hits = counts['tp']
    false_alarms = counts['fp']
    misses = counts['fn']
    true_negs = counts['tn']    

    h = hits/(hits+misses)
    f = false_alarms/(false_alarms+true_negs)
    
    a = np.log(f)-np.log(h)+np.log(1-h)-np.log(1-f)
    b = np.log(f)+np.log(h)+np.log(1-h)+np.log(1-f)
    return a/b
    

def get_sedi_et(counts, counts_random):
    sedi = get_sedi(counts)
    sedi_random = get_sedi(counts_random)
    #print('sedi:',sedi)
    #print('sedi_random:',sedi_random)
    if np.isnan(sedi_random): 
        sedi_random = 0.0 
    #print('sedi_et:',(sedi - sedi_random)/(1 - sedi_random))
    if np.isnan(sedi): 
        return 0.0
    else: 
        return (sedi - sedi_random)/(1 - sedi_random)
    #return sedi
    
    
def get_counts_from_binary_per_time(predictions, targets, args):    
    cm = np.zeros((args.frames_predict, 2, 2))
    
#     for t in range(0,args.frames_predict):
#         mat = confusion_matrix(targets[:,t].reshape(-1), predictions[:,t].reshape(-1), labels=[0,1])
#         cm[t,:,:] = mat[:,:]
             
#     print(cm)
   
    correct = targets==predictions 
    false = targets!=predictions
    
    for t in range(0, args.frames_predict):
        for i,j in zip([0,1],[1,0]):
            ci = targets[:,t]==i # where categories are i 
            cm[t, j, j] += (ci & correct[:,t]).sum()
            
            b1 = ci & false[:,t]              #prediction was wrong
            b2 = predictions[:,t]==j            #predicted category was j 
            cm[t, j, i] += (b1 & b2).sum() 
                      
    counts = {t: {'tp':0,'fp':0, 'fn':0, 'tn':0} for t in range(0, args.frames_predict)}
        
    for t in range(0,args.frames_predict):
        counts[t]['tp'] = cm[t,0,0]
        counts[t]['fp'] = cm[t,:,0].sum() - cm[t,0,0]
        counts[t]['fn'] = cm[t,0,:].sum() - cm[t,0,0]
        counts[t]['tn'] = cm[t,:,:].sum() - counts[t]['tp'] - counts[t]['fp'] - counts[t]['fn']
    
    return counts


def get_counts_from_binary(predictions, targets):    
    
    #cm = confusion_matrix(targets.reshape(-1), predictions.reshape(-1), labels=[0,1])
    #print(cm)
    
    cm = np.zeros((2, 2))
    
    correct = targets==predictions 
    false = targets!=predictions
    
    for i,j in zip([0,1],[1,0]):
        #print(i,t,(pred_cats[:,t]==i).sum()/(target_cats[:,t]==i).sum())
        ci = targets==i # where categories are i 
        cm[j, j] += (ci & correct).sum()

        b1 = ci & false              #prediction was wrong
        b2 = predictions==j            #predicted category was j 
        cm[j, i] += (b1 & b2).sum()     
    
    counts = {'tp':0,'fp':0, 'fn':0, 'tn':0}

    counts['tp'] = cm[0,0]
    counts['fp'] = cm[:,0].sum() - cm[0,0]
    counts['fn'] = cm[0,:].sum() - cm[0,0]
    counts['tn'] = cm[:,:].sum() - counts['tp'] - counts['fp'] - counts['fn']
   
    return counts


####################################################################
#                           Verification                           #
####################################################################


def minimum_coverage(preds, preds_random, targs, args, scale=(1,1), threshold=0, time_comp=True): 
    # data dim = (Num, t, l, l)
    offset = int((scale[0]-1)/2)
    s = scale[0]*scale[1]
    n = targs.shape[-1]
    N = (n-2*offset)**2
    
    reduced_rands = np.zeros(shape=(preds_random.shape[0], preds_random.shape[1], n-2*offset, n-2*offset))
    reduced_preds = np.zeros(shape=(preds.shape[0], preds.shape[1], n-2*offset, n-2*offset))
    reduced_targs = np.zeros(shape=(targs.shape[0], targs.shape[1], n-2*offset, n-2*offset))
    
    if scale == (1,1): 
        reduced_rands = preds_random
        reduced_preds = preds >= threshold
        reduced_targs = targs >= threshold 
    else: 
        for i in range(offset, n-offset):
            for j in range(offset, n-offset):  
                #if threshold>=0:
                Rj = (preds_random[:,:,i-offset:i+1+offset,j-offset:j+1+offset]).sum(axis=(2,3))/s
                Pj = (preds[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>=threshold).sum(axis=(2,3))/s
                Tj = (targs[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>=threshold).sum(axis=(2,3))/s
                #if threshold<0:
                #    Rj = (preds_random[:,:,i-offset:i+1+offset,j-offset:j+1+offset]).sum(axis=(2,3))/s
                #    Pj = (preds[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
                #    Tj = (targs[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
                    
                # minimum coverage: 
                reduced_rands[:,:,i-offset,j-offset] = Rj >= 1/s
                reduced_preds[:,:,i-offset,j-offset] = Pj >= 1/s
                reduced_targs[:,:,i-offset,j-offset] = Tj >= 1/s 
       
    #print('\n')
    #print(reduced_rands.sum()/(reduced_rands.shape[0]*reduced_rands.shape[1]*reduced_rands.shape[2]*reduced_rands.shape[3]))
    counts = get_counts_from_binary(reduced_preds, reduced_targs)
    counts_random = get_counts_from_binary(reduced_rands, reduced_targs)
    scores_scales = get_sedi_et(counts, counts_random)
    #print(counts_random)
    #print('\n')
    
    counts = get_counts_from_binary_per_time(reduced_preds, reduced_targs, args)
    counts_random = get_counts_from_binary_per_time(reduced_rands, reduced_targs, args)
    scores_times = np.array([get_sedi_et(counts[t], counts_random[t]) for t in counts.keys()])

    return scores_scales, scores_times 
   

def fractions_skill_score(pred, targets, scale=(1,1), threshold=None, categorical=False, time_comp=True):
    # data dim = (Num, t, l, l)
    offset = int((scale[0]-1)/2)
    s = scale[0]*scale[1]
    n = targets.shape[-1]
    N = (n-2*offset)**2
    FBS = np.zeros(shape=(targets.shape[0],targets.shape[1]))
    FBS_worst = np.zeros(shape=(targets.shape[0],targets.shape[1]))
    for i in range(offset, n-offset):
        for j in range(offset, n-offset):
            if categorical: 
                Pj = (np.floor(pred[:,:,i-offset:i+1+offset,j-offset:j+1+offset])==threshold).sum(axis=(2,3))/s
                Tj = (np.floor(targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset])==threshold).sum(axis=(2,3))/s
            else:        
                if threshold>0:
                    Pj = (pred[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>threshold).sum(axis=(2,3))/s
                    Tj = (targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>threshold).sum(axis=(2,3))/s
                if threshold<0:
                    Pj = (pred[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
                    Tj = (targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
            FBS += (Pj - Tj)**2
            FBS_worst += (Pj**2 + Tj**2)
    
    if time_comp==False:
        FBS = FBS.sum(1)
        FBS_worst = FBS_worst.sum(1) 
    
    FBS = FBS.sum(0)
    FBS_worst = FBS_worst.sum(0) 
    fss = 1. - FBS/FBS_worst
    return fss


def compute_IS(targets, preds, thresholds, args, categorical=False, time_comp=True):
    # number of scales: 7 (1,2,4,8,16,32,64)
    
    if time_comp: 
        time_tot = args.frames_predict 
        SS_time = []
    else: 
        time_tot = 1
        
    counts = {time:{t:{} for t in thresholds} for time in range(time_tot)}
        
    for time in range(time_tot): 
    
        E_t = np.zeros((len(thresholds),7))
        E_p = np.zeros((len(thresholds),7))
        MSE = np.zeros((len(thresholds),7))
        MSE_rand = np.zeros(len(thresholds))
        Et = np.zeros((len(thresholds)))
        Ep = np.zeros((len(thresholds)))
        
        if time_comp: 
            targ = targets[:,time].reshape(-1,64,64)
            pred = preds[:,time].reshape(-1,64,64)
        else: 
            targ = targets.reshape(-1,64,64)
            pred = preds.reshape(-1,64,64)
        if categorical: 
            targ = np.floor(targ)
            pred = np.floor(pred)
        
        n = len(targ)
        
        for k, t in enumerate(thresholds): 
            for frame_t, frame_p in zip(targ, pred):
                
                #counts = get_counts_from_binary(frame_p>t, frame_t>t) # for r and s 
                
                ct = pywt.wavedec2(frame_t>t, wavelet='haar')
                cp = pywt.wavedec2(frame_p>t, wavelet='haar')

                for i in range(7):
                    ct_copy, cp_copy = ct.copy(), cp.copy()
                    for j in range(7):
                        if j!=i:
                            ct_copy[j] = tuple([np.zeros_like(v) for v in ct_copy[j]])
                            cp_copy[j] = tuple([np.zeros_like(v) for v in cp_copy[j]])
                    
                    
                    rec_t = pywt.waverec2(ct_copy, wavelet='haar')
                    rec_p = pywt.waverec2(cp_copy, wavelet='haar')

                    Et[k] += (rec_t**2).mean()
                    Ep[k] += (rec_p**2).mean()
                    MSE[k][i] += ((rec_t-rec_p)**2).mean()
            
                #tot = counts['tp']+counts['fp']+counts['fn']+counts['tn'] # a + b + c + d 
                #r = (counts['tp']+counts['fp'])/tot # (a + b)/n
                #s = (counts['tp']+counts['fn'])/tot # (a + c)/n
                #B = r/s  
                #MSE_rand[k] = (B*s*(1-s)+s*(1-B*s))/7
            
            Ep[k]/=n
            Et[k]/=n
            B = Ep[k]/Et[k]
            MSE_rand[k] = (B*Et[k]*(1-Et[k])+Et[k]*(1-B*Et[k]))/7
            MSE[k,:]/=n
                
        #E_t/=n
        #E_p/=n
        #MSE/=n
        #MSE_rand/=n

        #r = E_p.sum(1)  #summed over all scales 
        #s = E_t.sum(1)

        SS = np.array([1-MSE[k,:]/MSE_rand[k] for k,t in enumerate(thresholds)])
        if not time_comp: 
            return SS
        
        SS_time.append(SS)
    return np.array(SS_time)

