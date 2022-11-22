import numpy as np
import pandas as pd 
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import cartopy.crs as ccrs
from sewar.full_ref import scc
import time
import torch
from PIL import Image
import gc 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns


import os
#os.chdir("./ConvLSTM_PyTorch_master/")
import sys 
sys.path.append("ConvLSTM_PyTorch_master/")

from data_loader import pchip, check_slopes, relevance_function

class args():
    def __init__(self):
        self.frames_predict=12
        self.batch_size=16
        self.hpa=1000
        self.num_years=42
        self.begin_testset=40
args = args()

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
        
    vmin, vmax = 0, data[indx].max()    
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

    vmin, vmax = 0, data0.max()    
    im0 = axs[0].imshow(data0, cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar0 = plt.colorbar(im0,ax=axs[0],fraction=0.046, pad=0.04)
    #axs[0].set_title('Maximum')
    
    vmin, vmax = 0, data1.max()
    im1 = axs[1].imshow(data1, cmap=cmap, interpolation='lanczos', vmin=vmin, vmax=vmax, extent=extent, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(im1,ax=axs[1],fraction=0.046, pad=0.04)
    #axs[1].set_title('Mean')
    if vmin==vmax:
        cbar1.set_ticks([0])
    
    vmin, vmax = 0, data2.max()
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
    
    x=np.arange(50,99,0.1)
    y=np.arange(1,50,0.1)
    ax.plot(x,y, linewidth=1.5, color='steelblue')

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
    ax.text(s='linear',x=61, y=25,fontsize=12)#r'$w=p-49'$
    ax.text(s='inverse',x=72, y=6,fontsize=12)#r'$w=50 \cdot \frac{1}{100-p}$'
    
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


def plot_forecast_comparison(inputs, targets, pred_list, date, t, vmin, vmax, label, ticklabels, save_as):
    plt.style.use('default')
    cmap = 'jet'
    cmap = matplotlib.cm.get_cmap('jet', 7)
    
    title_list = ['Input',
                  'Target',
                  r'W-MAE$_{inv}$',
                  r'W-MSE$_{inv}$',
                  r'W-MAE$_{lin}$',
                  r'W-MSE$_{lin}$',
                  r'SERA$_{p90}$',
                  r'SERA$_{p75}$',
                  r'SERA$_{p50}$',
                  'MAE',
                  'MSE',
                  #'Ensemble'
                 ]
    
    rows = len(title_list)
    gs_top = plt.GridSpec(nrows=rows, ncols=12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)
    gs_base = plt.GridSpec(nrows=rows, ncols=12, hspace=0.25, wspace=-0.7, left=0.1, right=0.9)

    fig, big_axes = plt.subplots(figsize=(24, 2*rows), nrows=rows, ncols=1, sharey=True)

    for i, big_ax in enumerate(big_axes):
        big_ax.set_title(title_list[i], fontsize=14, x=0.04, y=[0.3,0.24,0.16,0.30,0.45,0.60,0.80,0.95,1.1,1.35,1.55][i], rotation=90)
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
        
    for a in range(2,rows): 
        for i in range(1+12*a,1+12*(a+1)):
            ax = fig.add_subplot(gs_base[a,i-(1+12*a)])
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(pred_list[a-2][i-(1+12*a)], cmap=cmap, vmin=vmin, vmax=vmax)
        
            box = ax.get_position()
            box.y0 = box.y0 + 0.
            box.y1 = box.y1 + 0.+0.018*(a-2)
            ax.set_position(box)
    
#     for i in range(1+12*(rows-1),1+12*rows):
#         ax = fig.add_subplot(gs_base[rows-1,i-(1+12*(rows-1))])
#         ax.set_aspect('equal')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         ax.imshow(pred_list[-1][i-(1+12*(rows-1))], cmap=cmap, vmin=vmin, vmax=vmax)

#         box = ax.get_position()
#         box.y0 = box.y0 + 0.
#         box.y1 = box.y1 + 0.14
#         ax.set_position(box)
    
    
    cbar_ax = fig.add_axes([0.845, 0.43, 0.008, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label, fontsize=14, rotation=90, labelpad=10)
    cbar.set_ticks([3/7,9/7,15/7,21/7,27/7,33/7,39/7,45/7])#range(vmin,vmax+1))
    cbar.set_ticklabels(ticklabels)
    
    plt.suptitle(str(date.date())+' '+str(date.time())[:5], fontsize=14, y=0.92)
    
    if save_as!=None: 
        plt.savefig(save_as, dpi=300, bbox_inches='tight',pad_inches=0)
        
    #plt.show()
    plt.close()

    def convertImage():
        img = Image.open(save_as)
#         img = img.convert("RGBA")
        
        width, height = img.size

        # Setting the points for cropped image
        left = 0
        top = 0
        right = width
        bottom = 0.95*height

        # Cropped image of above dimension
        # (It will not change original image)
        img = img.crop((left, top, right, bottom))

#         datas = img.getdata()

#         newData = []

#         for item in datas:
#             if item[0] == 255 and item[1] == 255 and item[2] == 255:
#                 newData.append((255, 255, 255, 0))
#             else:
#                 newData.append(item)

#         img.putdata(newData)

        img.save(save_as, "PNG")
        #print("Successful")

    convertImage()
    
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
    plt.style.use('default')

    model_names = [['wmae_i_4',r'WMAE$_{inv}$ (4)'],
                   ['wmse_i_4','WMSE$_{inv}$ (4)'],
                   ['wmae_l_5','WMAE$_{lin}$ (5)'],
                   ['wmse_l_4','WMSE$_{lin}$ (4)'],
                   ['sera_p90_5','SERA$_{p90}$ (5)'],
                   ['sera_p75_5','SERA$_{p75}$ (5)'],
                   ['sera_p50_5','SERA$_{p50}$ (5)'],
                   ['mae_5','MAE (5)'],
                   ['mse_5','MSE (5)'],
                   ['persistence','Persistence']]

    markers = ['o','v','s','D','^','<','>','P','X']
        
    fig, big_axes = plt.subplots(1,1,figsize=(10,10))
    big_axes._frameon = False
    big_axes.get_xaxis().set_ticks([])
    big_axes.get_yaxis().set_ticks([])
    big_axes.get_xaxis().set_ticklabels([])
    big_axes.get_yaxis().set_ticklabels([])
    big_axes.set_xlabel('Lead-time [h]', fontsize=14, labelpad=40)
    
    for i, score_name, score_label in [[1,r'$H$','pod'],[2,'FAR','far'],[4,'TS','csi'],[5,'$B$','bias'],[6,'RMSE','rmse']]:
        
        ax = fig.add_subplot(2,3,i)
            
        #ax.set_xlabel('Lead-time [h]', fontsize=14, labelpad=10)
        #ax.set_ylabel(score_name, fontsize=14, labelpad=10)
        ax.set_title(score_name, fontsize=14)
    
        for a, (name, label) in enumerate(model_names):

            scores = np.load(os.path.join(root,'%s/scores_times.p'%name),allow_pickle=True)[score_label] #dim:(p,s,t)
            
            if score_name=='RMSE':
                scores = scores[-5,:]
            else:
                scores = scores[-5,0,:] # 99th percentile and smallest scale 0 
            
            ax.grid(True, alpha=0.5)
            ax.set_xticks(np.arange(2,14,2))
            if i>2: 
                ax.set_xticklabels(np.arange(2,14,2))
            else:
                ax.set_xticklabels([])

            if a == 9:
                ax.plot(np.arange(1,13), scores, c='black', marker='.', markersize=6, ls=':', linewidth=1.5, label=label)
            else:
                ax.plot(np.arange(1,13), scores, ls=':', marker=markers[a], markersize=6, linewidth=1.5, label=label)

        ax.tick_params(axis='both', which='major', labelsize=12)
        
        if i==2: 
            ax.legend(loc=0, fontsize=12, fancybox=True, bbox_to_anchor=(1.35, 0.9), frameon=True)
   
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.1)

    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 


def plot_model_comparison_all(root, save_as):
    plt.style.use('default')

    model_names = [['wmae_i_4',r'WMAE$_{inv}$ (4)'],
                   ['wmse_i_4','WMSE$_{inv}$ (4)'],
                   ['wmae_l_5','WMAE$_{lin}$ (5)'],
                   ['wmse_l_4','WMSE$_{lin}$ (4)'],
                   ['sera_p90_5','SERA$_{p90}$ (5)'],
                   ['sera_p75_5','SERA$_{p75}$ (5)'],
                   ['sera_p50_5','SERA$_{p50}$ (5)'],
                   ['mae_5','MAE (5)'],
                   ['mse_5','MSE (5)'],
                   ['persistence','Persistence']]

    markers = ['o','v','s','D','^','<','>','P','X']
    
    gs_top = plt.GridSpec(nrows=5, ncols=4, hspace=0.05, wspace=0.2, left=0.06, right=0.87, top=0.96)
    gs_base = plt.GridSpec(nrows=5, ncols=4, hspace=0.05, wspace=0.2, left=0.06, right=0.87, top=0.96)
        
    fig, big_axes = plt.subplots(figsize=(16,16), nrows=5, ncols=1, sharey=True)

    for i, big_ax in enumerate(big_axes):
        #big_ax.set_ylabel([r'$H$','FAR','TS',r'$B$','RMSE'][i], fontsize=14, labelpad=70)
        big_ax._frameon = False
        big_ax.get_xaxis().set_ticks([])
        big_ax.get_yaxis().set_ticks([])
        big_ax.get_xaxis().set_ticklabels([])
        big_ax.get_yaxis().set_ticklabels([])
        if i==4:
            big_ax.set_xlabel('Lead-time [h]', fontsize=14, labelpad=60, x=0.44)
    
    for i, (score_name, score_label) in enumerate([[r'$H$','pod'],['FAR','far'],['TS','csi'],['$B$','bias'],['RMSE','rmse']]):
        
        for j, threshold in enumerate([r'$f,o \geq p_{75}$',r'$f,o \geq p_{90}$',r'$f,o \geq p_{95}$',r'$f,o \geq p_{99}$']):
            
            
            if i<4:
                ax = fig.add_subplot(gs_top[i, j])
                
            else: 
                ax = fig.add_subplot(gs_base[i, j])
                box = ax.get_position()
                box.y0 = box.y0 + 0.-0.025
                box.y1 = box.y1 + 0.-0.025
                ax.set_position(box)
            
            if j==0: 
                ax.set_ylabel([r'$H$','FAR','TS',r'$B$','RMSE'][i], fontsize=14, labelpad=10)
            
            #ax = fig.add_subplot(5,4,(j+1)+4*(i-1))
            
            #ax.set_xlabel('Lead-time [h]', fontsize=14, labelpad=10)
            #ax.set_ylabel(score_name, fontsize=14, labelpad=10)
            if i==0:
                ax.set_title(threshold, fontsize=14, pad=15)
            if i==4: 
                ax.set_title([r'$p_{75} \leq o < p_{90}$',
                              r'$p_{90} \leq o < p_{95}$',
                              r'$p_{95} \leq o < p_{99}$',
                              r'$p_{99} \leq o < p_{99.9}$'][j], fontsize=14, pad=15)
             
            #if j==0:
            #    ax.set_ylabel(score_name, fontsize=14, labelpad=10)
                
            ax.grid(True, alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xticks(np.arange(2,14,2))
            
            if i==4: 
                ax.set_xticklabels(np.arange(2,14,2))
            else:
                ax.set_xticklabels([])

            for a, (name, label) in enumerate(model_names):

                scores = np.load(os.path.join(root,'%s/scores_times.p'%name),allow_pickle=True)[score_label] #dim:(p,s,t)

                if score_name=='RMSE':
                    scores = scores[j+1,:]
                else:
                    scores = scores[j+1,0,:] # smallest scale 0 

                if a == 9:
                    ax.plot(np.arange(1,13), scores, c='black', marker='.', markersize=6, ls=':', linewidth=1.5, label=label)
                else:
                    ax.plot(np.arange(1,13), scores, ls=':', marker=markers[a], markersize=6, linewidth=1.5, label=label)

            if (i==0) and (j==3): 
                ax.legend(loc=0, fontsize=12, fancybox=True, bbox_to_anchor=(1.05, 0.9), frameon=True)

    
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.18, hspace=0.25)
    #plt.tight_layout()
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 

def plot_model_comparison_scales(root, save_as):
    plt.style.use('seaborn')

    model_names = [['wmae_i_4',r'WMAE$_{inv}$ (4)'],
                   ['wmse_i_4','WMSE$_{inv}$ (4)'],
                   ['wmae_l_5','WMAE$_{lin}$ (5)'],
                   ['wmse_l_4','WMSE$_{lin}$ (4)'],
                   ['sera_p90_5','SERA$_{p90}$ (5)'],
                   ['sera_p75_5','SERA$_{p75}$ (5)'],
                   ['sera_p50_5','SERA$_{p50}$ (5)'],
                   ['mae_5','MAE (5)'],
                   ['mse_5','MSE (5)'],
                   ['persistence','Persistence']]

    markers = ['o','v','s','D','^','<','>','P','X']
   
    NUM_COLORS=len(model_names)-1
    #sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('hls',n_colors=NUM_COLORS)  # a list of RGB tuples
    colors = plt.cm.Dark2(np.linspace(0, 1, NUM_COLORS))
        
    fig, big_axes = plt.subplots(1,1,figsize=(12,6))
    big_axes._frameon = False
    big_axes.get_xaxis().set_ticks([])
    big_axes.get_yaxis().set_ticks([])
    big_axes.get_xaxis().set_ticklabels([])
    big_axes.get_yaxis().set_ticklabels([])
    big_axes.set_xlabel('Spatial scale [km]', fontsize=14, labelpad=40)
    
    for i, (score_name, score_label) in enumerate([[r'$H$','pod'],['FAR','far'],['TS','csi'],[r'$B$','bias']]):
        
        ax = fig.add_subplot(1,4,i+1)
            
        #ax.set_xlabel('Lead-time [h]', fontsize=14, labelpad=10)
        #ax.set_ylabel(score_name, fontsize=14, labelpad=10)
        ax.set_title(score_name, fontsize=14, pad=10)
    
        for a, (name, label) in enumerate(model_names):

            scores = np.load(os.path.join(root,'%s/scores_times.p'%name),allow_pickle=True)[score_label] #dim:(p,s)
            
            scores = scores[-2,:,-1] # 99th percentile and smallest scale 0 

            ax.set_xticks(np.arange(1,6))
            ax.set_xticklabels(np.array(np.round(np.array([1,3,5,7,9])*111/4,0),dtype=int))

            if a == 9:
                ax.plot(np.arange(1,6), scores, c='black', marker='.', markersize=6, ls=':', linewidth=1.5, label=label)
            else:
                ax.plot(np.arange(1,6), scores, ls=':', marker=markers[a], markersize=6, linewidth=1.5, label=label)

        ax.tick_params(axis='both', which='major', labelsize=12)
        
        if i==3: 
            ax.legend(loc=0, fontsize=12, fancybox=True, bbox_to_anchor=(1.1, 0.9), frameon=True)
    
   
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

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
               vmin=0.50, 
               vmax=1.0,
               cbar=True,
               cbar_kws={'ticks':[0.5,1.0],'shrink':0.6}) 

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
    
    
def forecasts_augmented(net, inputs, targets, device):   
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
            
            pred = net(torch.FloatTensor(inputs[i]).unsqueeze(0).unsqueeze(2).to(device)).squeeze().detach().cpu().numpy()
            pred_ = net(torch.FloatTensor(inputs_[i]).unsqueeze(0).unsqueeze(2).to(device)).squeeze().detach().cpu().numpy()
            
            list_preds.append(pred) 
            list_preds_.append(pred_) 
            
        preds = np.array(list_preds)  
        preds_ = np.array(list_preds_)  

        a = np.sqrt(np.mean((targets-preds)**2))
        b = np.sqrt(np.mean((targets-preds_)**2))
        rmse[t] = (1-a/b)*100

    return rmse



def plot_score_over_leadtime(root, save_as): 
    plt.style.use('default')
    
    labels = [[r'WMAE$_{inv}$',r'WMSE$_{inv}$'], [r'WMAE$_{lin}$',r'WMSE$_{lin}$'], [r'SERA$_{p90}$',r'SERA$_{p75}$',r'SERA$_{p50}$'], ['MAE','MSE']]
    
    titles = ['Inversely weighted','Linearly weighted','SERA','Standard']
    
    markers = [['o','v'],['s','D'],['^','<','>'],['P','X']]
    
    fig, big_axes = plt.subplots(1,1,figsize=(12,8),sharex=True,sharey=True)
    big_axes._frameon = False
    big_axes.get_xaxis().set_ticks([])
    big_axes.get_yaxis().set_ticks([])
    big_axes.get_xaxis().set_ticklabels([])
    big_axes.get_yaxis().set_ticklabels([])
    big_axes.set_xlabel('Input frame T [h]', fontsize=14, labelpad=30)
    big_axes.set_ylabel('RMSE skill score [%]', fontsize=14, labelpad=30)
    
    for i, losses in enumerate([['wmae_i','wmse_i'],['wmae_l','wmse_l'],['sera_p90','sera_p75','sera_p50'],['mae','mse']]):
        
        
        if i > 0:
            ax = fig.add_subplot(2,2,i+1, sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(2,2,i+1)
            
        ax.set_title(titles[i], pad=10, fontsize=14)
        
        #ax.set_ylim(0.,2.)
        #ax.set_yticks(np.arange(0.,2.0+0.2,0.2))
        ax.grid(True)
        ax.set_xticks(range(1,13))
        if i>1:
            ax.set_xticklabels(range(-11,1))
        else:
            ax.set_xticklabels([])
        
        
        for j, loss in enumerate(losses):
            label = labels[i][j]
            rmse = np.load(os.path.join(root,'rmse_array_%s.npy'%loss))            
            ax.plot(range(1,13), rmse, ls=':', markersize=6, linewidth=1.5, marker=markers[i][j], label=labels[i][j])
            
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 

def plot_distributions(root, save_as): 
    
    def outline(x, y):
        xs, ys = [], []
        for i in range(len(x)):
            xs.append(x[i])
            xs.append(x[i] + 0.5)
            ys.append(y[i])
            ys.append(y[i])
        return xs, ys
    
    
    plt.style.use('default')
    
    labels = [[r'WMAE$_{inv}$',r'WMSE$_{inv}$'], [r'WMAE$_{lin}$',r'WMSE$_{lin}$'], [r'SERA$_{p90}$',r'SERA$_{p75}$',r'SERA$_{p50}$'], ['MAE','MSE']]
    
    titles = ['Inversely weighted','Linearly weighted','SERA','Standard']
    
    markers = [['o','v'],['s','D'],['^','<','>'],['P','X']]
    linestyles = ['-.','--','-']
    
    mn,mx=-3.5,3.5
    
    fig, big_axes = plt.subplots(1,1,figsize=(12,8),sharex=True,sharey=True)
    big_axes._frameon = False
    big_axes.get_xaxis().set_ticks([])
    big_axes.get_yaxis().set_ticks([])
    big_axes.get_xaxis().set_ticklabels([])
    big_axes.get_yaxis().set_ticklabels([])
    big_axes.set_xlabel('Standardised wind speed', fontsize=14, labelpad=30)
    big_axes.set_ylabel('Frequency', fontsize=14, labelpad=40)
    
  
    for i, losses in enumerate([['wmae_i','wmse_i'],['wmae_l','wmse_l'],['sera_p90','sera_p75','sera_p50'],['mae','mse']]):
    #for i, losses in enumerate([['wmae_5_1000_40years','wmse_5_1000_40years'],['sera_5_1000_40years_2'],['mae_4_1000_40years'],['mse_5_1000_40years']]):   
        
        
        
        if i > 0:
            ax = fig.add_subplot(2,2,i+1, sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(2,2,i+1)
            
        ax.set_title(titles[i], pad=10, fontsize=14)
        
        #ax.set_ylim(0.,2.)
        #ax.set_yticks(np.arange(0.,2.0+0.2,0.2))
        ax.xaxis.grid(which='major')
        ax.xaxis.grid(which='minor')
        #ax.set_yscale('log')
        
        ax.xaxis.set_major_locator(MultipleLocator(1.))
        ax.xaxis.set_major_formatter('{x:.0f}')
        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        
#         ax.set_xticks(np.arange(mn,mx+1,0.5))
#         if i>1:
#             ax.set_xticklabels(np.arange(mn,mx+1))
#         else:
#             ax.set_xticklabels([])

        ax.set_xlim(xmin=mn,xmax=mx)
    
        for j, loss in enumerate(losses):
            bins, hist = np.load(os.path.join(root,'dist_array_%s.npy'%loss))            
            #ax.plot(bins, hist, ls=linestyles[j], label=labels[i][j])
            ax.bar(bins+j*0.5/len(losses), hist, width=0.5/len(losses), linewidth=0, alpha=0.9, edgecolor='black', label=labels[i][j], align='edge')
            
        bins, hist = np.load(os.path.join(root,'dist_array_target.npy'))      
        #ax.plot(bins, hist, c='black', ls=':', label='Target')
        #ax.bar(bins, hist, width=0.5, linewidth=1.5, color='black',alpha=0.6, edgecolor='black', label='Target', align='edge')
        bins, hist = outline(bins, hist)
        ax.plot(bins, hist, c='black', linewidth=1, ls='--', label='Target')
        
        

            
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_as!=None: 
        plt.savefig(save_as)
    plt.show()
    plt.close()
    return 


    


