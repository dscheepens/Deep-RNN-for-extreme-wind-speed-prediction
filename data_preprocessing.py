import numpy as np 
import matplotlib.pyplot as plt 

def standardize_local(data):
    zvals = []
    for i in range(data.shape[1]):
        m = data[:,i].mean(0)
        s = data[:,i].std(0)
        zvals.append([m,s])
        for j in range(len(data)):
            data[j,i] = (data[j,i] - m)/s
    return data, zvals
    
# def standardize_climate(data):
#     plt.figure(figsize=(30,10))
#     for yr in range(42):
#         plt.scatter(np.arange(365*8), data[yr*365*8:(yr+1)*365*8,0,0], marker='o', c='blue', alpha=0.05)
#     plt.savefig('era5/climatology.png')
#     plt.close()
    
#     n = data.shape[0]
#     zvals = []
#     for i in range(365*24):
#         indices = range(i,n,365*8)
#         m = data[indices].mean(0)
#         s = data[indices].std(0)
#         zvals.append([m,s])
#         data[indices] = (data[indices] - m)/s

#     plt.figure(figsize=(30,10))
#     for yr in range(42):
#         plt.scatter(np.arange(365*8), data[yr*365*8:(yr+1)*365*8,0,0], marker='o', c='blue', alpha=0.05)
#     plt.savefig('era5/climatology_standardized.png')
#     plt.close()
#     return data, zvals


if __name__ == '__main__':
    
    root = '../../../../../mnt/data/scheepensd94dm/data/'
    
    #inputs = np.load(root + 'adaptor.mars.internal-Horizontal_velocity.npy')[:24*365*10] #take first 10 years 
    
    inputs = np.load(root + 'adaptor.mars.internal-Horizontal_velocity_850.npy')
    
    print('inputs shape:',inputs.shape) # shape: (N, 64, 64)
    print(inputs.min(), inputs.max())
    
    # standardize
    inputs, zvals = standardize_local(inputs)
    np.save(root + 'zvals_global.npy', zvals)
    
    print(inputs.min(), inputs.max())
    print(inputs.mean(), inputs.std())
    
    targets = inputs
    
    # divide into (12,64,64) chunks 
    num_samples = len(inputs)
    
    num_input = 12
    num_output = 12
    samples_per_chunk = num_input + num_output 
    
    from_ = 0
    to_ = num_samples - samples_per_chunk
    shift_ = 12 
    
    chunked_inputs = []
    for i in range(from_, to_, shift_):
        chunked = []
        for chunk in inputs[i:i+num_input]:
            chunked.append(chunk)
        chunked_inputs.append(chunked)
    inputs = np.array(chunked_inputs, dtype=float)   
    print('\ninputs shape', inputs.shape)
    print(inputs.min(),inputs.max())
    np.save(root + 'chunked_inputs.npy', inputs) 
    del inputs, chunked_inputs
    
    chunked_targets = []
    for i in range(from_, to_, shift_):
        chunked = []
        for chunk in targets[i+num_input:i+num_input+num_output]:
            chunked.append(chunk)
        chunked_targets.append(chunked)
    targets = np.array(chunked_targets, dtype=float)
    print('\ntargets shape',targets.shape)
    print(targets.min(),targets.max())
    np.save(root + 'chunked_targets.npy', targets) 
    del chunked_targets
    
    categorized = np.array(np.floor(targets),dtype=int)  
    del targets 
    
    