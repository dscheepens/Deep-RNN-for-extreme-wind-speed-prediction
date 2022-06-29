import numpy as np
import os

def preprocessing(args):
    
    root = '../../../../../mnt/data/scheepensd94dm/data/'
    
    data = np.load(os.path.join(root, 'adaptor.mars.internal-Horizontal_velocity_%s.npy'%args.hpa))[:24*365*args.num_years]
    print('number of years:', len(data)/(24*365))
    m = data.mean(0)
    s = data.std(0)
    for j in range(len(data)):
        data[j] = (data[j] - m)/s
        
    np.save(os.path.join(root, 'era5_standardised.npy'), data)
    return 
        
        
if __name__ == "__main__":
    
    class args():
        def __init__(self):
            self.hpa=1000
            self.num_years=42
            
    args = args()
    preprocessing(args)