# Deep-RNN-for-extreme-wind-speed-prediction
Paper code for "A deep convolutional RNN model for spatio-temporal prediction of wind speed extremes in the short-to-medium range for wind energy applications"

## Data 

Wind speed data was obtained from the Copernicus Climate Data Store: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form. 

Used were 'U-component of wind' and 'V-component of wind' at pressure levels 1000, 925, 850 and 775 hPa (seperately) with an hourly interval for a 10 year duration and between 3-18.75 longitude, 40-55.75 latitude. 

As it is, data_loader.py requires the data to be in the .npy format, rather than .grib. This can be achieved with the file grib_to_numpy.py.
