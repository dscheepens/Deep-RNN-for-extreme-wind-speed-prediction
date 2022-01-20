import torch
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np 
from joblib import Parallel, delayed
import time 


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


####################################################################
#                 Data analysis related functions                  #
####################################################################


def clip(arr,l,r): 
    arr[arr<l]=l
    arr[arr>=r]=r-1e-6
    return arr 


def categorize(grid):
    res = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            coord = grid[i][j]
            for l in range(-3,12):  
                if coord >= l and coord < l+1: 
                    res[i][j]=l
                    break
    return res 


def standardize_local(data):
    data2 = data.copy()
    m = data2.mean(0)
    s = data2.std(0)
    zvals = [m,s]
    #for j in range(len(data)):
    data2 = (data2 - m)/s
    return data2, zvals


def standardize_climate(data):
    data2 = data.copy()
    zvals = []
    for i in range(365*24):
        indices = range(i,data2.shape[0],365*24)
        m = data2[indices].mean(0)
        s = data2[indices].std(0)
        zvals.append([m,s])
        data2[indices] = (data2[indices] - m)/s
    return data2, zvals


def get_climatology(years=40):   
    data = np.load('../era5/adaptor.mars.internal-Horizontal_velocity.npy')[:365*24*years]
    data, zvals = standardize_local(data)
    
    n = data.shape[0]
    
    means = []
     
    for i in range(365*24):
        indices = range(i,n,365*24)
        m = data[indices].mean(0)
        means.append(m)
    
    
    n_test = int((0.2*n)/(365*24)) 
    climatology = np.array(means*n_test).reshape(-1,12,64,64) 
    return climatology


def get_persistence(inputs):
    persist = inputs.copy()
    for i in range(inputs.shape[0]):
        last_frame = inputs[i,-1]
        for j in range(inputs.shape[1]):
            persist[i,j] = last_frame
    return persist 

    
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

def get_counts_per_cat(predictions, targets):
    cm = np.zeros((r-l,r-l))
    
    target_cats = np.floor(targets)
    pred_cats = np.floor(predictions)
        
    correct = target_cats==pred_cats 
    false = target_cats!=pred_cats
    
    for i in range(l, r):
        ci = target_cats==i # where categories are i 
        cm[i-l, i-l] += (ci & correct).sum()
        for j in range(l, r):
            if j != i: 
                b1 = ci & false              #prediction was wrong
                b2 = pred_cats==j            #predicted category was j 
                cm[i-l,j-l] += (b1 & b2).sum() 
     
    #cm = confusion_matrix(target_cats.reshape(-1), pred_cats.reshape(-1), labels=list(range(-2,4)))
    
    counts = {c: {'tp':0,'fp':0, 'fn':0, 'tn':0} for c in range(l,r)}
        
    for i in range(0,r-l):
        counts[i+l]['tp'] = cm[i,i]
        counts[i+l]['fp'] = cm[:,i].sum() - cm[i,i]
        counts[i+l]['fn'] = cm[i,:].sum() - cm[i,i]
        counts[i+l]['tn'] = cm[:,:].sum() - counts[i+l]['tp'] - counts[i+l]['fp'] - counts[i+l]['fn']
    
    return counts


def accuracy_per_cat(dataloader, model, device, args):
    model.eval()
    inputs = []
    predictions = []
    targets = []
    cats = []
    
    for batch in dataloader:  
        batch_data = batch[0].to(device).unsqueeze(2)
        d = batch_data.shape[-1]
        pred = model(batch_data).detach().cpu().numpy()
        target = batch[1].detach().cpu().numpy()
        cat = batch[2].detach().cpu().numpy()
        
        inputs.append(batch_data.squeeze().detach().cpu().numpy())
        predictions.append(pred)
        targets.append(target)
        cats.append(cat)

    inputs = clip(np.concatenate(inputs, axis=0),l,r)
    predictions = clip(np.concatenate(predictions, axis=0),l,r).reshape(-1,d,d)
    targets = clip(np.concatenate(targets, axis=0),l,r).reshape(-1,d,d)
    cats = clip(np.concatenate(cats, axis=0),l,r).reshape(-1,d,d)
        
    counts = get_counts_per_cat(predictions, targets)
    
    scores = {c: 
               get_accuracy_scores(counts[c])
              for c in counts.keys()} 

    # counts with climatology prediction
    predictions = get_climatology().reshape(-1,d,d)
    counts_climate = get_counts_per_cat(predictions, targets)
    
    # counts with persistence prediction
    predictions = get_persistence(inputs).reshape(-1,d,d)
    counts_persist = get_counts_per_cat(predictions, targets)
    
    skill_scores = {c: 
                     get_gilbert_skill_scores(counts[c], counts_climate[c], counts_persist[c]) 
                    for c in counts.keys()}    
    
    return scores, skill_scores


def error_per_cat(testloader, model, device):
    model.eval()
    
    me = {x: 0 for x in range(l, r)}
    bias = {x: 0 for x in range(l, r)}
    mae = {x: 0 for x in range(l, r)}
    cc = {x: 0 for x in range(l, r)}
    ac_climate = {x: 0 for x in range(l, r)}
    ac_persist = {x: 0 for x in range(l, r)}
    
    inputs = []
    predictions = []
    targets = [] 

    for batch in testloader:  
        batch_data = batch[0].to(device).unsqueeze(2)
        pred = model(batch_data).detach().cpu().numpy()
        target = batch[1].detach().cpu().numpy()
        cat = batch[2].detach().cpu().numpy()
        
        inputs.append(batch_data.squeeze().detach().cpu().numpy())
        predictions.append(pred)
        targets.append(target)
    
    inputs = clip(np.concatenate(inputs, axis=0),l,r)
    predictions = clip(np.concatenate(predictions, axis=0),l,r)
    targets = clip(np.concatenate(targets, axis=0),l,r)
    cats = np.floor(targets)
    
    #naiv = mase_naiv(trainloader, args)
    pred_climate = get_climatology() 
    pred_persist = get_persistence(inputs)    
    
    for i in range(l, r):
        ci = cats==i
        if ci.sum()>0:

            p = predictions[ci]
            t = targets[ci]
            error = p-t
            n = ci.sum()

            me[i] = error.sum()/n
            bias[i] = p.sum()/t.sum()
            mae[i] = abs(error).sum()/n
            rmse[i] = np.sqrt(error**2).sum()/n
            #mase[i] = abs(error).sum()/n/naiv[i][j]

            cc[i] = (p*t).sum()/np.sqrt((p**2).sum()*(t**2).sum()+1e-9)

            f = p-pred_climate[ci]
            a = t-pred_climate[ci]
            ac_climate[i] = (f*a).sum()/np.sqrt((f**2).sum()*(a**2).sum()+1e-9)

            f = p-pred_persist[ci]
            a = t-pred_persist[ci]
            ac_persist[i] = (f*a).sum()/np.sqrt((f**2).sum()*(a**2).sum()+1e-9)
    
    return me, bias, mae, cc, ac_climate, ac_persist


def get_counts_per_cat_per_time(predictions, targets, args):
        
    cm = np.zeros((args.frames_predict, r-l, r-l))
    
    target_cats = np.floor(targets)
    pred_cats = np.floor(predictions)
    
    
#     for t in range(0,args.frames_predict):
#         print(t)
#         target_cats[:,t]
#         mat = confusion_matrix(target_cats[:,t].reshape(-1), pred_cats[:,t].reshape(-1), labels=list(range(-2,4)))
#         print(mat.shape)
#         print(mat)
#         cm[t,:,:] = mat[:,:]
    
    correct = target_cats==pred_cats 
    false = target_cats!=pred_cats
    
    for i in range(l, r):
        for t in range(0, args.frames_predict):
            ci = target_cats[:,t]==i # where categories are i 
            cm[t, i-l, i-l] += (ci & correct[:,t]).sum()
            for j in range(l, r):
                if j != i: 
                    b1 = ci & false[:,t]              #prediction was wrong
                    b2 = pred_cats[:,t]==j            #predicted category was j 
                    cm[t, i-l,j-l] += (b1 & b2).sum() 
              
    counts = {c: {t: {'tp':0,'fp':0, 'fn':0, 'tn':0} for t in range(0, args.frames_predict)} for c in range(l,r)}
        
    for i in range(0,r-l):
        for t in range(0,args.frames_predict):
            counts[i+l][t]['tp'] = cm[t,i,i]
            counts[i+l][t]['fp'] = cm[t,:,i].sum() - cm[t,i,i]
            counts[i+l][t]['fn'] = cm[t,i,:].sum() - cm[t,i,i]
            counts[i+l][t]['tn'] = cm[t,:,:].sum() - counts[i+l][t]['tp'] - counts[i+l][t]['fp'] - counts[i+l][t]['fn']
    
    return counts
        
    
def sedi_score(h,f): 
    a = np.log(f)-np.log(h)+np.log(1-h)-np.log(1-f)
    b = np.log(f)+np.log(h)+np.log(1-h)+np.log(1-f)
    return a/b


def get_accuracy_scores(counts): 
    hits = counts['tp']
    false_alarms = counts['fp']
    misses = counts['fn']
    true_negs = counts['tn']    
    
    #bias = (hits+false_alarms)/(hits+misses)
    H = hits/(hits+misses)
    F = false_alarms/(false_alarms+true_negs)
    far = false_alarms/(hits+false_alarms)
    pre = hits/(hits+false_alarms)
    f1 = 2*(pre*H)/(pre+H)
    ts = hits/(hits+misses+false_alarms)
    pss = H-F
    css = (hits*true_negs-false_alarms*misses)/((hits+false_alarms)*(misses+true_negs))
    sedi = sedi_score(H,F)
    
    return np.array([sedi, pss, css, H, pre, f1])
   
    
def get_gilbert_skill_scores(counts, counts_climate, counts_persist): 
    hits, misses, false_alarms, correct_neg = counts['tp'], counts['fn'], counts['fp'], counts['tn']

    # random     
    tot = hits+misses+false_alarms+correct_neg
    hits_random = (hits+misses)*(hits+false_alarms)/tot
    GSS_random = (hits-hits_random)/(hits+misses+false_alarms-hits_random)

    # climatology 
    hits_climate = counts_climate['tp']
    GSS_climate = (hits-hits_climate)/(hits+misses+false_alarms-hits_climate)
    
    # persistence 
    hits_persist = counts_persist['tp']
    GSS_persist = (hits-hits_persist)/(hits+misses+false_alarms-hits_persist)
    
    return [GSS_random, GSS_climate, GSS_persist]

def accuracy_per_cat_per_time(dataloader, model, device, args):
    model.eval()
    inputs = []
    predictions = []
    targets = []
    cats = []
    
    for batch in dataloader:  
        batch_data = batch[0].to(device).unsqueeze(2)
        pred = model(batch_data).detach().cpu().numpy()
        target = batch[1].detach().cpu().numpy()
        cat = batch[2].detach().cpu().numpy()
        
        inputs.append(batch_data.squeeze().detach().cpu().numpy())
        predictions.append(pred)
        targets.append(target)
        cats.append(cat)

    inputs = clip(np.concatenate(inputs, axis=0),l,r)
    predictions = clip(np.concatenate(predictions, axis=0),l,r)
    targets = clip(np.concatenate(targets, axis=0),l,r)
    cats = clip(np.concatenate(cats, axis=0),l,r)
        
    counts = get_counts_per_cat_per_time(predictions, targets, args)
       
    scores = {c: 
              {t: 
               get_accuracy_scores(counts[c][t])
               for t in counts[c].keys()} 
              for c in counts.keys()} 
    
    # counts with climatology prediction
    predictions = get_climatology()
    counts_climate = get_counts_per_cat_per_time(predictions, targets, args)
    
    # counts with persistence prediction
    predictions = get_persistence(inputs)
    counts_persist = get_counts_per_cat_per_time(predictions, targets, args)
    
    skill_scores = {c: 
                    {t: 
                     get_gilbert_skill_scores(counts[c][t], counts_climate[c][t], counts_persist[c][t])
                     for t in counts[c].keys()} 
                    for c in counts.keys()}
    
    return scores, skill_scores 


def mase_naiv(trainloader, args):
    
    naiv = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}

    inputs = []
    cats = []
    for batch in trainloader: 
        batch_data = batch[0].detach().cpu().numpy()
        target = batch[1].detach().cpu().numpy()
        cat = batch[2].detach().cpu().numpy()
        
        batch_data[batch_data<l]=l
        batch_data[batch_data>=r]=r-1e-6
        
        cat[cat<l]=l
        cat[cat>=r]=r-1
        
        inputs.append(batch_data)
        cats.append(cat)
    
    inputs = clip(np.concatenate(inputs, axis=0),l,r)
    cats = clip(np.concatenate(cats, axis=0),l,r)
    
    predictions = get_persistence(inputs)
    
    for i in range(l, r):
        for t in range(0, args.frames_predict): 
            #start = 12 if t==0 else t  
            #stop = -1 if t==0 else None
            #print(inputs[start:stop:12].shape, inputs[start-1:stop:12].shape)
            #ci = cats[start:stop:12]==i
            ci = cats[:,t]==i
            if ci.sum()>0:
                #naiv[i][t] = abs(inputs[start:stop:12][ci] - inputs[start-1:stop:12][ci]).sum()/ci.sum()
                naiv[i][t] = abs(predictions[:,t][ci] - inputs[:,t][ci]).sum()/ci.sum()
                
    return naiv

def error_per_cat_per_time(trainloader, testloader, model, device, args):
    model.eval()
    
    me = {c: {t: 0.0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    bias = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    mae = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    #rmse = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    mase = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    cc = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    ac_climate = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    ac_persist = {c: {t: 0 for t in range(0, args.frames_predict)} for c in range(l, r)}
    
    inputs = []
    predictions = []
    targets = [] 
    cats = [] 
    for batch in testloader:  
        batch_data = batch[0].to(device).unsqueeze(2)
        pred = model(batch_data).detach().cpu().numpy()
        target = batch[1].detach().cpu().numpy()
        cat = batch[2].detach().cpu().numpy()
        
        inputs.append(batch_data.squeeze().detach().cpu().numpy())
        predictions.append(pred)
        targets.append(target)
        cats.append(cat)
    
    inputs = clip(np.concatenate(inputs, axis=0),l,r)
    predictions = clip(np.concatenate(predictions, axis=0),l,r)
    targets = clip(np.concatenate(targets, axis=0),l,r)
    cats = np.floor(targets)
    
    #naiv = mase_naiv(trainloader, args)
    pred_climate = get_climatology() 
    pred_persist = get_persistence(inputs)    
    
    for i in range(l, r):
        for j in range(0, args.frames_predict): 
            ci = cats[:,j]==i
            if ci.sum()>0:
                
                p = predictions[:,j][ci]
                t = targets[:,j][ci]
                error = p-t
                n = ci.sum()

                me[i][j] = error.sum()/n
                bias[i][j] = p.sum()/t.sum()
                mae[i][j] = abs(error).sum()/n
                #mse[i][j] = (error**2).sum()/n
                #mase[i][j] = abs(error).sum()/n/naiv[i][j]
                
                cc[i][j] = (p*t).sum()/np.sqrt((p**2).sum()*(t**2).sum()+1e-9)
                
                f = p-pred_climate[:,j][ci]
                a = t-pred_climate[:,j][ci]
                ac_climate[i][j] = (f*a).sum()/np.sqrt((f**2).sum()*(a**2).sum()+1e-9)
                
                f = p-pred_persist[:,j][ci]
                a = t-pred_persist[:,j][ci]
                ac_persist[i][j] = (f*a).sum()/np.sqrt((f**2).sum()*(a**2).sum()+1e-9)
    
    return me, bias, mae, cc, ac_climate, ac_persist



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

