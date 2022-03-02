import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, TensorDataset, WeightedRandomSampler, RandomSampler
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import utilities.utils as utils 

def get_percentiles(data):
    
    l, r = data.min(), data.max()
    r = 8
    num=500
    
    n, bins = utils.function_hist(data, l, r, num)
    
    cs = np.cumsum(n)
    
    loc90 = np.where(cs>0.90*sum(n))[0][0]*(r-l)/num+l
    loc99 = np.where(cs>0.99*sum(n))[0][0]*(r-l)/num+l
    
    return np.round(loc90,1), np.round(loc99,1)
    

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
    
    
def relevance(x, y1, y2):
    points = np.array([[y1,0.,0.],[y2,1.,0.]])
    a,b,c,d = pchip(points)
    rel = relevance_function(x.numpy(),y1,y2,a,b[0],c,d)
    return torch.from_numpy(rel).float()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        r = self.tensors[2][index]

        return x, y, r

    def __len__(self):
        return self.tensors[0].size(0) 
    

def chunkify(data, which=None):
    num_samples = len(data)

    num_input = 12
    num_output = 12
    samples_per_chunk = num_input + num_output 

    from_ = 0
    to_ = num_samples - samples_per_chunk
    shift_ = 12 

    chunked_data = []
    for i in range(from_, to_, shift_):
        chunked = []
        if which=='inputs': 
            for chunk in data[i : i+num_input]:
                chunked.append(chunk)
        if which=='targets':
            for chunk in data[i+num_input : i+num_input+num_output]:
                chunked.append(chunk)
        chunked_data.append(chunked)
        
    return np.array(chunked_data, dtype=float)   


    
def load_era5(root, args, a, b, c):
    # Load ERA5 train, validation and test data. 
    
    #root = '../../../../../../mnt/data/scheepensd94dm/data/'
    
    
    # get 90th and 99th percentiles from all training and validation data: 
    data = np.load(os.path.join(root, 'adaptor.mars.internal-Horizontal_velocity_%s.npy'%args.hpa)[:24*365*8])
    data = utils.standardize_local(data)[0]
    
    y1, y2 = get_percentiles(data)
    print('percentiles:',y1, y2) 
    
    #inputs = np.load(os.path.join(root, 'chunked_inputs.npy'))
    inputs = chunkify(data, which='inputs')
    n = len(inputs)
    
    train_inputs = torch.from_numpy(np.concatenate((inputs[:int(n*a)], inputs[int(n*b):int(n*c)]))).float()
    val_inputs = torch.from_numpy(inputs[int(n*a):int(n*b)]).float()
    test_inputs = torch.from_numpy(inputs[int(n*c):]).float()
    del inputs
    
    #targets = np.load(os.path.join(root, 'chunked_targets.npy'))
    targets = chunkify(data, which='targets')
    train_targets = torch.from_numpy(np.concatenate((targets[:int(n*a)], targets[int(n*b):int(n*c)]))).float()
    val_targets = torch.from_numpy(targets[int(n*a):int(n*b)]).float()
    test_targets = torch.from_numpy(targets[int(n*c):]).float()
    del targets
    
    train_rels = relevance(train_targets, y1, y2)
    val_rels = relevance(val_targets, y1, y2)
    test_rels = relevance(test_targets, y1, y2)    
    
    #transform=transforms.Compose([
    #    transforms.ToTensor(),
    #    AddGaussianNoise(0., 1.)])
    
    #train_dataset = CustomTensorDataset(tensors=(train_inputs, train_targets, train_rels), 
    #                                    transform=AddGaussianNoise(0., 0.1))
   
    train_dataset = TensorDataset(*(train_inputs, train_targets, train_rels))
    del train_inputs, train_targets, train_rels
    val_dataset = TensorDataset(*(val_inputs, val_targets, val_rels))
    del val_inputs, val_targets, val_rels
    test_dataset = TensorDataset(*(test_inputs, test_targets, test_rels))
    del test_inputs, test_targets, test_rels

    #sampler = WeightedRandomSampler(train_rels+0.01, int(len(train_rels)/2), replacement=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         #sampler = sampler, 
                                         drop_last=False)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         drop_last=False)

    return train_loader, valid_loader, test_loader
    
    
def get_storm_data(data):
    threshold = 4.
    new_storm = False
    no_storm = True
    last_storm_index = 0
    last_nonstorm_index = 0
    sliding_window_count = 0
    unique_storm_count = 0
    post_storm = False
    post_storm_wait = 0

    storm_chunks = []
    nonstorm_chunks = []

    for i, grid in enumerate(data):

        if new_storm == True:
            storm_chunks.append(data[i-31:i+1])
            sliding_window_count += 1
            if sliding_window_count == 16: 
                # reset
                new_storm = False
                sliding_window_count = 0
                post_storm = True 
                post_storm_wait = 0
            continue 

        if post_storm == True: 
            # wait for 1 day 
            post_storm_wait += 1
            if post_storm_wait >= 8:
                post_storm = False
                if grid.max() >= threshold: 
                    last_nonstorm_index = i+1
                else: 
                    last_nonstorm_index = i 
            continue 

        if new_storm == False:
            if grid.max() >= threshold:
                last_nonstorm_index = i+1 # earlierst possible
                if i - last_storm_index >= 32: 
                    new_storm = True
                    storm_chunks.append(data[i-31:i+1])
                    sliding_window_count += 1
                    last_storm_index = i 
                    unique_storm_count += 1
            else: 
                if i - last_nonstorm_index >= 32: 
                    nonstorm_chunks.append(data[i-31:i+1])

    return storm_chunks, nonstorm_chunks


