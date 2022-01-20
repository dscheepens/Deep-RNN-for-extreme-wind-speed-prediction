import numpy as np 
import utils 
import pywt 

    
def minimum_coverage(predictions, targets, args, scale=(1,1), threshold=None, categorical=False, time_comp=True): 
    # data dim = (Num, t, l, l)
    offset = int((scale[0]-1)/2)
    s = scale[0]*scale[1]
    n = targets.shape[-1]
    N = (n-2*offset)**2
    
    reduced_preds = np.zeros(shape=(predictions.shape[0], predictions.shape[1], n-2*offset, n-2*offset))
    reduced_targs = np.zeros(shape=(targets.shape[0], targets.shape[1], n-2*offset, n-2*offset))
    
    if scale == (1,1): 
        reduced_preds = predictions > threshold
        reduced_targs = targets > threshold 
    else: 
        for i in range(offset, n-offset):
            for j in range(offset, n-offset):  
                if categorical: 
                    Pj = (np.floor(predictions[:,:,i-offset:i+1+offset,j-offset:j+1+offset])==threshold).sum(axis=(2,3))/s
                    Tj = (np.floor(targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset])==threshold).sum(axis=(2,3))/s
                else:        
                    if threshold>=0:
                        Pj = (predictions[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>threshold).sum(axis=(2,3))/s
                        Tj = (targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset]>threshold).sum(axis=(2,3))/s
                    if threshold<0:
                        Pj = (predictions[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
                        Tj = (targets[:,:,i-offset:i+1+offset,j-offset:j+1+offset]<threshold).sum(axis=(2,3))/s
                    
                # minimum coverage: 
                reduced_preds[:,:,i-offset,j-offset] = Pj >= 1/s
                reduced_targs[:,:,i-offset,j-offset] = Tj >= 1/s 
       
    if time_comp: 
        counts = utils.get_counts_from_binary_per_time(reduced_preds, reduced_targs, args)
        scores = np.array([utils.get_accuracy_scores(counts[t]) for t in counts.keys()])
    else: 
        counts = utils.get_counts_from_binary(reduced_preds, reduced_targs)
        scores = utils.get_accuracy_scores(counts)
    return scores
   

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

