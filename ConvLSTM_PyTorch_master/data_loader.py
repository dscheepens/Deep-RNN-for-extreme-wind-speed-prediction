import math
import numpy as np
import os
import random
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, TensorDataset, WeightedRandomSampler, RandomSampler
import matplotlib.pyplot as plt 
#import torchvision.transforms as transforms
import utils
import gc 


def get_sera_percentiles(data, p0, p1):
    percentiles = []
    for p in [p0,p1]:
        percentiles.append(np.percentile(data,p))
    return np.array(percentiles)
    

def pchip(points):
    h = np.zeros((len(points)-1))
    d = np.zeros((len(points)-1))
    a = np.zeros((len(points)-1))
    c = np.zeros((len(points)-1))
    
    for k in range(len(points)-1):
        h[k] = points[k+1][0] - points[k][0]
        d[k] = (points[k+1][1] - points[k][1])/h[k]
        a[k] = points[k][1]
        
    b = check_slopes(points[:,2], d)
    for k in range(len(points)-1):
        c[k] = (3*d[k]-2*b[k]+b[k+1])/h[k]
        d[k] = (b[k]-2*d[k]+b[k+1])/h[k]**2
        
    return a, b, c, d
    
    
def check_slopes(deriv, delta): 
    x = deriv.copy()
    y = delta.copy()
    
    for k in range(len(x)-1): 
        if y[k]==0: 
            x[k], x[k+1] = 0, 0
        else: 
            alpha = x[k]/y[k]
            beta = x[k+1]/y[k]
            if (x[k]!=0) and (alpha<0):
                x[k] = -x[k]
                alpha = x[k]/y[k]
            if (x[k+1]!=0) and (beta<0):
                x[k+1] = -x[k+1]
                beta = x[k+1]/y[k]
            tau1 = 2*alpha+beta-3 
            tau2 = alpha+2*beta-3
            if (tau1>0) and (tau2>0) and (alpha*(tau1+tau2)<tau1*tau2): 
                tau = 3*y[k]/np.sqrt(alpha**2+beta**2)
                x[k] = alpha*tau
                x[k+1] = beta*tau
    return x


def relevance_function(arr,y1,y2,a,b,c,d):
    rel = np.zeros(arr.shape)
    tf = (arr>=y1) & (arr<y2)
    rel[tf] = a+b*(arr[tf]-y1)+c*(arr[tf]-y1)**2+d*(arr[tf]-y1)**3
    rel[arr>=y2] = arr[arr>=y2]>=y2
    return rel
    
    
def calc_relevance(data):
    y1, y2 = get_sera_percentiles(data)
    print('y1, y2:',np.round(y1,2),np.round(y2,2))
    points = np.array([[y1,0.,0.],[y2,1.,0.]])
    a,b,c,d = pchip(points)
    rel = relevance_function(data,y1,y2,a,b[0],c,d)
    return rel #torch.from_numpy(rel).float()


def calc_relevance_local_percentiles(data, p0=90, p1=99):
    relevances = np.zeros(data.shape)
    for i in range(64):
        for j in range(64):                 
            y1, y2 = get_sera_percentiles(data[:,i,j], p0, p1)
            #print('y1, y2:',np.round(y1,2),np.round(y2,2))
            points = np.array([[y1,0.,0.],[y2,1.,0.]])
            a,b,c,d = pchip(points)
            relevances[:,i,j] = relevance_function(data[:,i,j],y1,y2,a,b[0],c,d)
            
    return relevances #torch.from_numpy(rel).float()


def calc_weights(data, args):
    l, r = int(np.floor(data.min())), int(np.floor(data.max()+1))
    n, bins = utils.function_hist(data, l, r, r+1-l)

    props = []
    props.append(sum(n[:abs(l)])/sum(n))

    count = abs(l)
    end = (abs(l)+r)
    while count < end: 
        props.append(n[count]/sum(n))
        if count==abs(l)+3: 
            props.append(sum(n[abs(l)+4:])/sum(n))
            break
        count+=1

    inverse = 1/np.array(props)
    weights = [float('%.2g' %(i/min(inverse))) for i in inverse]
    print('weights:',weights)
    args.weights = [weights[0] for i in range(abs(l)-1)] + weights + [weights[-1] for i in range(10)]
    args.offset = abs(l)
    return 
    
    
def calc_weights_local_percentiles(data, method='inv'):
    
    def get_percentiles(data):
        percentiles = []
        for p in np.arange(50,100):
            percentiles.append(np.percentile(data,p))
        return np.array(percentiles)
    
    if method=='inv':
        weights = 50/np.arange(50,0,-1)
        weights/=weights[0]
    elif method=='lin':
        weights = np.arange(1,51,1)
    
    all_weights = np.zeros(data.shape)
    for i in range(64):
        print(i)
        for j in range(64):

            percentiles = get_percentiles(data[:,i,j])
            percentiles = np.append(percentiles, percentiles[-1]+100)

            cats = np.zeros(data[:,i,j].shape, dtype=int)
            for k in range(len(percentiles)-1):
                indices = np.where( (data[:,i,j]>=percentiles[k]) & (data[:,i,j]<percentiles[k+1]) )
                cats[indices] = k

            all_weights[:,i,j] = weights[cats]    
            
    return all_weights 

from sys import getsizeof
    
def load_era5(root, args, a, b, c, training=True):
    ### Load ERA5 train, validation and test data. 
    
    print('loading data')
    data = np.load(root+'era5_standardised.npy')
    
    print('number of data years:', len(data)/(24*365))
           
    if training:   
        data = data[:24*365*c] # don't use test data
        print('number of training/validation years:', len(data)/(24*365))
        
        # supplementary for sera, wmae or wmse
        if args.loss == 'sera':
            sup = calc_relevance_local_percentiles(data, args.p0, args.p1) 
            print('relevances computed.')
            #chunkify:
            sup = sup[:int(sup.shape[0]/12)*12].reshape((int(sup.shape[0]/12),12,64,64))
        elif args.loss in ['wmae', 'wmse']: 
            sup = calc_weights_local_percentiles(data, method=args.weighting_method)
            print('weights computed.')
            #chunkify:
            sup = sup[:int(sup.shape[0]/12)*12].reshape((int(sup.shape[0]/12),12,64,64))
            
        print('done.')
        
        #chunkify: 
        data = data[:int(data.shape[0]/12)*12].reshape((int(data.shape[0]/12),12,64,64))
        # (inputs are given as data[:-1], targets as data[1:])
        n = len(data)
        
        print('data chunked.')
        gc.collect()
                
        data_train = data[:int(n*a)]
        data_val = data[int(n*a):]
        del data 
        gc.collect()
        print('done')
        
        train_inputs = torch.from_numpy(data_train[:-1]).float()
        print('done')
        train_targets = torch.from_numpy(data_train[1:]).float()
        del data_train 
        gc.collect()
        print('train part done.')

        val_inputs = torch.from_numpy(data_val[:-1]).float()
        print('done')
        val_targets = torch.from_numpy(data_val[1:]).float()
        del data_val
        gc.collect()
        print('val part done.')
                   
        if args.loss not in ['mae','mse']: 
            
            sup_train = sup[:int(n*a)]
            sup_val = sup[int(n*a):]
            del sup 
            gc.collect()
            
            train_sups = torch.from_numpy(sup_train[1:]).float()
            del sup_train
            gc.collect()
            print('1')
            
            val_sups = torch.from_numpy(sup_val[1:]).float()
            del sup_val
            gc.collect()
            print('2')
            
            print('preparing train dataset')
            
            train_dataset = TensorDataset(*(train_inputs, train_targets, train_sups))
            del train_inputs, train_targets, train_sups
            gc.collect()
            
            print('preparing val dataset')
            
            val_dataset = TensorDataset(*(val_inputs, val_targets, val_sups))
            del val_inputs, val_targets, val_sups
            gc.collect()
        else: 
            train_dataset = TensorDataset(*(train_inputs, train_targets, train_targets))
            val_dataset = TensorDataset(*(val_inputs, val_targets, val_targets))
            del train_inputs, train_targets, val_inputs, val_targets
            gc.collect()

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         drop_last=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         drop_last=True)
        
        return train_loader, valid_loader
        
    if not training:   
        
        data = data[24*365*c:24*365*args.num_years]
                
        #chunkify:
        data = data[:int(data.shape[0]/12)*12].reshape((int(data.shape[0]/12),12,64,64))
        
        print('converting to Tensors...')
        inputs = torch.FloatTensor(data[:-1])
        targets = torch.FloatTensor(data[1:])
        del data  
        gc.collect()
        print('done')  
        
        test_dataset = TensorDataset(*(inputs, targets))
        del inputs, targets
        gc.collect()
        print('done')
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         drop_last=False)
        
        return test_loader


