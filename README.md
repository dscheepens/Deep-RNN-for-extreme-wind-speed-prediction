# Deep-RNN-for-extreme-wind-speed-prediction
Paper code for "An adapted convolutional RNN model for spatio-temporal prediction of wind speed extremes in the short-to-medium range for wind energy applications"

Model and training code can be found in `\ConvLSTM_PyTorch_master`. 

Example model forecasts can be found in `\example_forecasts`.

All figures can be reconstructed in `visualisation_notebook.ipynb`, except for the forecast visualisations, which require that the models in question have been trained and saved using `\ConvLSTM_PyTorch_master/main.py`.   

All scores were computed with `save_scores.py` and have been saved in `\saved_scores`. 

Clone repository: 

```python
git clone https://github.com/dscheepens/Deep-RNN-for-extreme-wind-speed-prediction.git 
```

## Data 

Wind speed data was obtained from the Copernicus Climate Data Store: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form. 

Used were 'U-component of wind' and 'V-component of wind' at pressure levels 1000, 925, 850 and 775 hPa (seperately) with an hourly interval for a 10 year duration and between 3-18.75 longitude, 40-55.75 latitude. 

`data_loader.py` requires the data to be in the .npy format, rather than .grib. This can be achieved with the file `grib_to_numpy.py`.

`preprocessing.py` standardises the data and saves it as era5_standardised.npy into the specified data root. 

## Results

<img 
src="front_example.png"
/>

## Citation 

```python
```
