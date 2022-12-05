#%% load modules
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider
from coastsat import SDS_transects
#%%

def analyze_shoreline(output,transects,settings,plot=True):
    
    if plot:
        # plot the transects to make sure they are correct (origin landwards!)
        fig = plt.figure(figsize=[15,8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.title('Location of the transects')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            color=matplotlib.cm.jet(i/len(output['shorelines']))
            plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'),color=color)
        for i,key in enumerate(list(transects.keys())):
            plt.plot(transects[key][0,0],transects[key][0,1], 'bo', ms=5)
            plt.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
            plt.text(transects[key][0,0]-100, transects[key][0,1]+100, key,
                        va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))

    cross_distance = SDS_transects.compute_intersection(output, transects, settings) 
    
    if plot:
        maxi=[]
        for k in cross_distance.keys():
           maxi.append(np.nanmax(cross_distance[k]))
        maxi=np.max(maxi)
        
        # Plot the new time series containing the prediction
        fig, ax = plt.subplots(figsize=[15,8])
        gs = gridspec.GridSpec(len(cross_distance),1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        ax.grid(linestyle=':', color='0.5')
        ax.set_ylim(0,maxi)
        ax.set_ylabel('Cross-distance (in metres)')
        l, = plt.plot(output['dates'], cross_distance['1'], '-o', ms=6, mfc='w')
        ax.margins(x=0)   
        
        axcolor = 'lightgray'
        axtrans = plt.axes([0.18, 0.025, 0.65, 0.04], facecolor=axcolor)      
        strans = Slider(axtrans, 'Transect', 1, len(list(cross_distance.keys())), valinit=1, valstep=1)
        
        def update(val):
            trans = strans.val
            l.set_ydata(cross_distance[str(trans)])
            fig.canvas.draw_idle()
        
        strans.on_changed(update)
        plt.title('Time series for each transect',pad=400)
        plt.show() 

    return cross_distance