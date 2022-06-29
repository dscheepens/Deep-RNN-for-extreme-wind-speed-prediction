import numpy as np
import pygrib 


# Data from: 
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
# 'U-component of wind' and 'V-component of wind' at 1000, 925, 850 or 775 hPa. 
# Files were renamed: adaptor.mars.internal-U-component_[hPa].grib


def grib_to_numpy(root, hpa): 
    arr = [] 
    
    # Merge U and V velocities
    grbs_u = pygrib.open(root+'adaptor.mars.internal-U-component_%s.grib'%hpa)
    grbs_v = pygrib.open(root+'adaptor.mars.internal-V-component_%s.grib'%hpa)

    for grb_u, grb_v in zip(grbs_u, grbs_v):
        values = np.sqrt(grb_u.values**2 + grb_v.values**2)
        arr.append(values)
        
    np.save(root+'adaptor.mars.internal-Horizontal_velocity_%s.npy'%hpa, np.array(arr))
    return 
    

if __name__ == "__main__":

    root = '../../../../../mnt/data/scheepensd94dm/data/'

    for hpa in [1000, 925, 850, 775]:
        grib_to_numpy(root, hpa)
    