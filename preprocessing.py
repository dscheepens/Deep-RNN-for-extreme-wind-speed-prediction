import numpy as np
import os
from sklearn.preprocessing import PowerTransformer

def preprocessing(args):
    
    root = '../../../../../mnt/data/scheepensd94dm/data/'
    
    data = np.load(os.path.join(root, 'adaptor.mars.internal-Horizontal_velocity_%s.npy'%args.hpa))[24*365*40:24*365*42]
    print('number of years:', len(data)/(24*365))

    for i in range(64):
        for j in range(64):
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            data[:,i,j] = pt.fit_transform(data[:,i,j].reshape(-1, 1)).squeeze()
    
#     m = data.mean(0)
#     s = data.std(0)
#     for j in range(len(data)):
#         data[j] = (data[j] - m)/s
        
    np.save(os.path.join(root, 'era5_standardised_test.npy'), data)
    return 
        
        
if __name__ == "__main__":
    
    class args():
        def __init__(self):
            self.hpa=1000
            self.num_years=40
            
    args = args()
    preprocessing(args)