import numpy as np
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import cartopy.crs as ccrs
from sewar.full_ref import scc
import time
import torch

import os
#os.chdir("./ConvLSTM_PyTorch_master/")
import sys 
sys.path.append("ConvLSTM_PyTorch_master/")

from data_loader import pchip, check_slopes, relevance_function



def plot_example_frame(data, indx, save_as):
    plt.style.use('default')
    
    fig = plt.figure(figsize=(10, 5))
    
    cmap = 'jet'
    
    central_lon, central_lat = 11, 48
    extent = [3, 18.75,40-0.15, 55.75+0.15]
    
    ax = fig.add_subplot(3,2,(2,6), projection=ccrs.Orthographic(central_lon, central_lat))
    
    ax.set_extent([2, 19.75, 39, 56.75])
    ax.gridlines(alpha=0.8)
    ax.coastlines(resolution='50m')
    
    ax.plot(2.75+0.25*12, 56-0.25*6, 's', color='black', markerfacecolor='white',markersize=7, transform=ccrs.PlateCarree())
    ax.plot(2.75+0.25*32, 56-0.25*32, 's', color='black',markerfacecolor='white',markersize=7, transform=ccrs.PlateCarree())
    ax.plot(2.75+0.25*56, 56-0.25*59, 's', color='black',markerfacecolor='white',markersize=7, transform=ccrs.PlateCarree())
        
    vmin, vmax = data[indx].min(), data[indx].max()    
    im0 = ax.imshow(data[indx], cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar0 = plt.colorbar(im0,ax=ax,fraction=0.046, pad=0.04)
    #cbar0.set_label(label=r'Wind speed [ms$^{-1}$]',size=14, labelpad=10)
    
    ax = fig.add_subplot(3,2,1)
    ax.plot(data[:24*30,12,6], linewidth=0.9)
    ax.axhline(data[:,12,6].mean(),ls='--',color='red', linewidth=0.9)
    ax.get_xaxis().set_visible(False)
    
    ax = fig.add_subplot(3,2,3)
    ax.plot(data[:24*30,32,32], linewidth=0.9)
    ax.axhline(data[:,32,32].mean(),ls='--',color='red', linewidth=0.9)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylabel('Wind speed [m/s]', fontsize=14)
    
    ax = fig.add_subplot(3,2,5)
    ax.plot(data[:24*30,56,59], linewidth=0.9)
    #ax.set_xticks(np.linspace(0,24*365,13))
    #ax.set_xticklabels(np.arange(1,14))
    ax.axhline(data[:,56,59].mean(),ls='--',color='red', linewidth=0.9)
    ax.get_xaxis().set_visible(False)
            
    plt.tight_layout() 
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return


def plot_maps(data, save_as):
    plt.style.use('default')
    
    cmap = 'jet'

    central_lon, central_lat = 11, 48
    extent = [3, 18.75,40, 55.75]

    fig, axs = plt.subplots(nrows=1,ncols=3,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            figsize=(16,12))

    axs=axs.flatten()

    for i in range(3):
        #ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))

        axs[i].set_extent(extent)
        #axs[i].gridlines(alpha=0.8)
        axs[i].coastlines(resolution='50m')
    
    data0 = data.max(0)
    data1 = data.mean(0)
    data2 = data.std(0)

    vmin, vmax = data0.min(), data0.max()    
    im0 = axs[0].imshow(data0, cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar0 = plt.colorbar(im0,ax=axs[0],fraction=0.046, pad=0.04)
    #axs[0].set_title('Maximum')
    
    vmin, vmax = data1.min(), data1.max()
    im1 = axs[1].imshow(data1, cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(im1,ax=axs[1],fraction=0.046, pad=0.04)
    #axs[1].set_title('Mean')
    if vmin==vmax:
        cbar1.set_ticks([0])
    
    vmin, vmax = data2.min(), data2.max()
    im2 = axs[2].imshow(data2, cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(im2,ax=axs[2],fraction=0.046, pad=0.04)
    #axs[2].set_title('Std')
    if vmin==vmax:
        cbar2.set_ticks([1])
        
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return


def plot_weight_function(save_as):
    plt.style.use('seaborn')

    fig,ax=plt.subplots(1,1,figsize=(6,4))
    x=np.arange(0,110,0.1)
    y=50/np.arange(50,0.9,-0.1)
    y/=y[0]
    y = np.append(np.ones(500),y)
    y = np.append(y, y[-1]*np.ones(109))

    ax.plot(x,y, linewidth=1.5,color='steelblue')

    ax.set_xticks([0,50,75,90,99,109])
    ax.set_xticklabels([0,50,75,90,99,100])

    ax.axvline(x=50,linewidth=1.,ls='--')

    ax.fill_between(x=[0,50],y1=[100,100],color='gray',alpha=0.05)
    ax.fill_between(x=[50,100],y1=[100,100],color='orange',alpha=0.05)

    ax.set_ylim(0,50)
    ax.set_xlim(0,99.1)

    ax.set_xlabel(r'Percentile $p$',fontsize=12)
    ax.set_ylabel(r'Weight $w$',fontsize=12)

    ax.text(s=r'$w=1$',x=22,y=34,fontsize=12)
    ax.text(s=r'$w=50 \cdot \frac{1}{100-p}$',x=63,y=34,fontsize=12)
    
    ax.grid(True)
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 



def plot_relevance_function(save_as):
    plt.style.use('seaborn')
    
    p90, p99 = 1, 2
    points = np.array([[p90,0,0],[p99,1,0]])
    a,b,c,d = pchip(points)
     
    values = np.linspace(0.5,2.5,100)
    relevances = relevance_function(values,points[0][0],points[1][0],a,b[0],c,d)

    fig, ax = plt.subplots(1,1,figsize=(6,4))

    ax.plot(values, relevances, linewidth=1.5)
    ax.axvline(x=points[0][0],ls='--',linewidth=1.)
    ax.axvline(x=points[1][0],ls='--',linewidth=1.)

    ax.set_xticks([p90,p99])
    ax.set_xticklabels([r'$p_{90}$',r'$p_{99}$'],fontsize=12)

    ax.set_xlabel('Percentile $p$',fontsize=12)
    ax.set_ylabel(r'Relevance $\phi$',fontsize=12)
    plt.tight_layout()
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 

def plot_forecast_percentiles(inputs, targets, preds, date, t, vmin, vmax, save_as):
    cmap = 'jet'
    
    title_list = ['Input',
                  'Target',
                  'Predicted']
    
    gs_top = plt.GridSpec(3, 12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)
    gs_base = plt.GridSpec(3, 12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)

    fig, big_axes = plt.subplots(figsize=(24, 6), nrows=3, ncols=1, sharey=True)

    for i, big_ax in enumerate(big_axes):
        big_ax.set_title(title_list[i], fontsize=14, x=0.04, y=[0.3,0.24,0.31][i], rotation=90)
        big_ax._frameon = False
        big_ax.get_xaxis().set_visible(False)
        big_ax.get_yaxis().set_visible(False)
    
    for i in range(1,13):
        ax = fig.add_subplot(gs_top[0,i-1])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(inputs[i-1], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=%s'%(i-12),x=0.5, y=1., fontsize=14)
    
    for i in range(13,25):
        ax = fig.add_subplot(gs_base[1,i-13])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(targets[i-13], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=+%s'%(i-12),x=0.5, y=1., fontsize=14)
        
    for i in range(25,37):
        ax = fig.add_subplot(gs_base[2,i-25])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(preds[i-25], cmap=cmap, vmin=vmin, vmax=vmax)
        
        box = ax.get_position()
        box.y0 = box.y0 + 0.
        box.y1 = box.y1 + 0.08
        ax.set_position(box)
    
    cbar_ax = fig.add_axes([0.845, 0.3, 0.008, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'threshold (percentile)', fontsize=14, rotation=90, labelpad=10)
    cbar.set_ticks([1,2,3,4,5,6])
    cbar.set_ticklabels([50,75,90,95,99,99.9])
    
    plt.suptitle("#%s | "%t+str(date.date())+' '+str(date.time())[:5], fontsize=14)
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return


def plot_forecast(inputs, targets, preds, date, t, vmin, vmax, save_as):
    cmap = 'jet'
    
    title_list = ['Input',
                  'Target',
                  'Predicted']
    
    gs_top = plt.GridSpec(3, 12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)
    gs_base = plt.GridSpec(3, 12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)

    fig, big_axes = plt.subplots(figsize=(24, 6), nrows=3, ncols=1, sharey=True)

    for i, big_ax in enumerate(big_axes):
        big_ax.set_title(title_list[i], fontsize=14, x=0.04, y=[0.3,0.24,0.31][i], rotation=90)
        big_ax._frameon = False
        big_ax.get_xaxis().set_visible(False)
        big_ax.get_yaxis().set_visible(False)
    
    for i in range(1,13):
        ax = fig.add_subplot(gs_top[0,i-1])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(inputs[i-1], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=%s'%(i-12),x=0.5, y=1., fontsize=14)
    
    for i in range(13,25):
        ax = fig.add_subplot(gs_base[1,i-13])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(targets[i-13], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=+%s'%(i-12),x=0.5, y=1., fontsize=14)
        
    for i in range(25,37):
        ax = fig.add_subplot(gs_base[2,i-25])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(preds[i-25], cmap=cmap, vmin=vmin, vmax=vmax)
        
        box = ax.get_position()
        box.y0 = box.y0 + 0.
        box.y1 = box.y1 + 0.08
        ax.set_position(box)
    
    cbar_ax = fig.add_axes([0.845, 0.3, 0.008, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'wind speed [$\sigma$]', fontsize=14, rotation=90, labelpad=10)
    cbar.set_ticks([0.0,1.0,2.0,3.0,4.0])
    
    plt.suptitle("#%s | "%t+str(date.date())+' '+str(date.time())[:5], fontsize=14)
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return


def plot_forecast_comparison(inputs, targets, pred_list, date, t, vmin, vmax, save_as):
    cmap = 'jet'
    
    title_list = ['Input',
                  'Target',
                  'W-MAE',
                  'W-MSE',
                  'SERA',
                  'MAE',
                  'MSE',
                  'Ensemble']
    
    gs_top = plt.GridSpec(nrows=8, ncols=12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)
    gs_base = plt.GridSpec(nrows=8, ncols=12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)

    fig, big_axes = plt.subplots(figsize=(24, 2*8), nrows=8, ncols=1, sharey=True)

    for i, big_ax in enumerate(big_axes):
        big_ax.set_title(title_list[i], fontsize=14, x=0.04, y=[0.3,0.24,0.22,0.40,0.65,0.88,1.05,0.83][i], rotation=90)
        big_ax._frameon = False
        big_ax.get_xaxis().set_visible(False)
        big_ax.get_yaxis().set_visible(False)
    
    for i in range(1,13):
        ax = fig.add_subplot(gs_top[0,i-1])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(inputs[i-1], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=%s'%(i-12),x=0.5, y=1., fontsize=14)
    
    for i in range(13,25):
        ax = fig.add_subplot(gs_base[1,i-13])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(targets[i-13], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title('T=+%s'%(i-12),x=0.5, y=1., fontsize=14)
        
    for a in range(2,7): 
        for i in range(1+12*a,1+12*(a+1)):
            ax = fig.add_subplot(gs_base[a,i-(1+12*a)])
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(pred_list[a-2][i-(1+12*a)], cmap=cmap, vmin=vmin, vmax=vmax)
        
            box = ax.get_position()
            box.y0 = box.y0 + 0.
            box.y1 = box.y1 + 0.+0.03*(a-2)
            ax.set_position(box)
    
    for i in range(1+12*7,1+12*8):
        ax = fig.add_subplot(gs_base[7,i-(1+12*7)])
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(pred_list[-1][i-(1+12*7)], cmap=cmap, vmin=vmin, vmax=vmax)

        box = ax.get_position()
        box.y0 = box.y0 + 0.
        box.y1 = box.y1 + 0.12
        ax.set_position(box)
    
    
    cbar_ax = fig.add_axes([0.845, 0.43, 0.008, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'wind speed [$\sigma$]', fontsize=14, rotation=90, labelpad=10)
    cbar.set_ticks([0.0,1.0,2.0,3.0,4.0])
    
    plt.suptitle(str(date.date())+' '+str(date.time())[:5], fontsize=14, y=0.93)
    if save_as!=None: 
        plt.savefig(save_as, dpi=300)
        
    plt.show()
    plt.close()
    return


def save_animation(inputs, targets, predictions, date, vmin, vmax, save_as):
    cmap='jet'
    
    num_images = len(inputs)+len(targets)
    fps = 3    
    dpi = 300
    
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.set_aspect('equal')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Input', fontsize=4, y=0.95)
    im1 = ax1.imshow(inputs[0], cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax2 = fig.add_subplot(132)
    ax2.set_aspect('equal')
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(0.5)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Target', fontsize=4, y=0.95)
    im2 = ax2.imshow(np.zeros((64,64)), cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax3 = fig.add_subplot(133)
    ax3.set_aspect('equal')
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(0.5)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title('Prediction', fontsize=4, y=0.95)
    im3 = ax3.imshow(np.zeros((64,64)), cmap=cmap, vmin=vmin, vmax=vmax)

    #im.set_clim([0,1])
    plt.suptitle(str(date.date())+' '+str(date.time())[:5], fontsize=4)
    fig.set_size_inches([2,1])
    
    plt.tight_layout()
    
    plt.subplots_adjust(left=0.01,
            bottom=0.01, 
            right=0.99, 
            top=0.99, 
            wspace=0.1, 
            hspace=0.1)
        
    def update_img(n):
        if n < len(inputs):
            tmp1 = inputs[n]
            im1.set_data(tmp1)
            fig.suptitle('T=%s'%(n-11), y=0.1, fontsize=4)
        else: 
            tmp1 = inputs[-1]
            im1.set_data(tmp1)
            tmp2 = targets[n-12]
            im2.set_data(tmp2)
            tmp3 = predictions[n-12]
            im3.set_data(tmp3)
            fig.suptitle('T=+%s'%(n-11), y=0.1, fontsize=4)
        return im1, im2, im3
    
    ani = animation.FuncAnimation(fig, update_img, num_images, interval=1000/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if save_as!=None: 
        ani.save(save_as, writer=writer, dpi=dpi)
    plt.close()
    return 


def plot_model_comparison_times(root, save_as):
    plt.style.use('seaborn')

    fig,ax = plt.subplots(1,1,figsize=(8,4))

    ax.set_xlabel('Lead-time [h]',fontsize=14,labelpad=10)
    ax.set_ylabel('SEDI',fontsize=14,labelpad=10)

    model_names = [['wmae_5_1000_40years','WMAE (5)'],
                   ['wmse_5_1000_40years','WMSE (5)'],
                   ['sera_5_1000_40years','SERA (5)'],
                   ['mae_4_1000_40years','MAE (4)'],
                   ['mse_5_1000_40years','MSE (5)'],
                   ['persistence','Persistence']]

    markers = ['o','s','D','^','<','.']
    
    for a, (name, label) in enumerate(model_names):
            
        scores_scales = np.load(os.path.join(root,'%s/scores_times.p'%name),allow_pickle=True)['sedi'] #dim:(p,s,t)
        
        scores = scores_scales[-2,0,:] # 99th percentile and smallest scale 0 
        
        ax.set_xticks(np.arange(1,13))
        ax.set_xticklabels(np.arange(1,13))

        if a == 5:
            ax.plot(np.arange(1,13), scores, c='grey', marker=markers[a], markersize=6, ls=':', linewidth=1.5, label=label)
        else:
            ax.plot(np.arange(1,13), scores, ls=':', marker=markers[a], markersize=6, linewidth=1.5, label=label)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc=0, fontsize=12, fancybox=True)
    plt.tight_layout()
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 
    
    
def plot_intensity_scale_diagram(root, model, save_as):
    plt.style.use('default')

    scores = np.load(os.path.join(root,'%s/scores_scales.p'%model),allow_pickle=True)['sedi']

    fig, ax = plt.subplots(1,1,figsize=(5, 4))    

    thresholds = [50, 75, 90, 95, 99, 99.9]
    scales = [int(i) for i in np.round(np.array([9,7,5,3,1])*111/4, 0)]

    axis_label = np.array(scales)
    df = pd.DataFrame(np.zeros((len(scales),len(thresholds))),index=scales, columns=thresholds, dtype=float)
    df = df.set_axis(axis_label, axis=0)
    for k, s in enumerate(scales): 
        df.iloc[-(k+1)] = scores[:,k]

    sb.heatmap(df, annot=True, fmt=".2f", 
               cmap='YlGnBu_r' ,
               vmin=0.65, 
               vmax=0.95,
               cbar=False) 

    ax.set_ylabel('Spatial scale [km]',fontsize=12,labelpad=10)
    ax.set_xlabel('Intensity threshold (percentile)',fontsize=12, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticklabels(axis_label, rotation=0, fontsize=12)
    ax.set_xticklabels(thresholds, fontsize=12)

    plt.tight_layout()
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 
    
    
def forecasts_augmented(ensemble, inputs, targets, device):   
    np.random.seed(999)
       
    rmse = np.zeros(12)      
        
    for t in range(12): 
        print('t=',t)
        random_indices = np.random.choice(np.arange(len(inputs)), size=len(inputs), replace=False)
        inputs_ = inputs.copy() 
        inputs_[:,t] = inputs_[random_indices][:,t] 
        
        list_preds = []
        list_preds_ = []
                                
        for i in range(len(inputs)):  
            pred = ensemble.predict(inputs[i], device)
            pred_ = ensemble.predict(inputs_[i], device)
            
            list_preds.append(pred) 
            list_preds_.append(pred_) 
            
        preds = np.array(list_preds)  
        preds_ = np.array(list_preds_)  

        a = np.sqrt(np.mean((targets-preds)**2))
        b = np.sqrt(np.mean((targets-preds_)**2))
        rmse[t] = (1-a/b)*100

    return rmse



def plot_score_over_leadtime(means, stds, ylabel, save_as): 
    plt.style.use('seaborn')

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    #ax.set_title('%s over lead-time'%ylabel, pad=10, fontsize=14)
    ax.plot(range(1,13),means,ls="--")
    ax.scatter(range(1,13),means)
    ax.set_ylabel(ylabel + ' skill score [%]', fontsize=12)
    ax.set_xlabel('Input frame T [h]',fontsize=12)
    #ax.set_ylim(0.,2.)
    #ax.set_yticks(np.arange(0.,2.0+0.2,0.2))
    ax.grid(True)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(range(-11,1))
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 





