#%% load modules
import os
import pickle
import numpy as np
from python import reconstruct_shoreline as rs
import matplotlib.pyplot as plt
#%%

#%% SUB-FUNCTIONS
# returns inf and sup of predicted time series using computed error
def cd_to_interval(time_series,mae,output,dates_pred):
    cd_low = dict([])
    cd_hig = dict([])
    ind = None
    find = True
    for trans in time_series.keys():
        cd_low[trans] = []
        cd_hig[trans] = []
        compt = 1
        for dat in range(len(time_series[trans])):
            err = 0
            if dates_pred[dat].date()>=output['dates'][-1].date():
                if find:
                    ind = dat
                    find = False
                try:
                    err = mae[trans][str(compt)]
                    compt += 1
                except:
                    err = mae[trans][str(compt-1)]
            cd_low[trans].append(time_series[trans][dat]-err)
            cd_hig[trans].append(time_series[trans][dat]+err)
        cd_low[trans] = np.array(cd_low[trans])
        cd_hig[trans] = np.array(cd_hig[trans])
    return cd_low, cd_hig, ind

# computes the area of a quadrangle given its vertices
def area_quadrangle(xA,yA,xB,yB,xC,yC,xD,yD):
    a = np.sqrt((xA-xB)**2+(yA-yB)**2)
    b = np.sqrt((xC-xB)**2+(yC-yB)**2)
    c = np.sqrt((xC-xD)**2+(yC-yD)**2)
    d = np.sqrt((xD-xA)**2+(yD-yA)**2)
    p = np.sqrt((xC-xA)**2+(yC-yA)**2)
    q = np.sqrt((xB-xD)**2+(yB-yD)**2)
    area = 0.25*np.sqrt(4*p**2*q**2-(b**2+d**2-a**2-c**2)**2)
    return area
#%%

# estimates the amount of population that is threatened by water rise/coastal erosion (in hectares)
def estimate_pop(cross_distance,predicted_sl,time_series_pred,dates_pred,n_months_further,transects,output,inputs,settings,model='Holt',density=None,plot=True): 
    # check if MAE has been computed 
    is_mae = False
    if os.path.exists(os.path.join(inputs['filepath'],inputs['sitename'],'mae_%s.pkl'%model)):
        f= open(os.path.join(inputs['filepath'],inputs['sitename'],'mae_%s.pkl'%model), 'rb') 
        mae = pickle.load(f)
        f.close()
        is_mae = True
    
    # get lower and upper boundaries of predicted cross-distances if MAE has been computed
    if is_mae:
        cd_low, cd_hig, ind = cd_to_interval(time_series_pred,mae,output,dates_pred)
        predicted_low = rs.reconstruct_shoreline(cd_low,transects,dates_pred,output,inputs,settings,n_months_further,estimate_pop=True)
        predicted_hig = rs.reconstruct_shoreline(cd_hig,transects,dates_pred,output,inputs,settings,n_months_further,estimate_pop=True)
    
    # get shoreline of current year (year of reference)
    current_year = cross_distance.copy()
    datprev = [output['dates'][-1]]
    for k in current_year.keys():
        current_year[k] = np.array([time_series_pred[k][ind]])
    current_year = rs.reconstruct_shoreline(current_year,transects,datprev,output,inputs,settings,1,estimate_pop=True)
    
    current = []
    for k in current_year.keys():
        current.append(current_year[k][0])
    current = np.array(current)
    
    areas_hig = np.zeros(len(list(predicted_sl.keys())))
    areas_mean = np.zeros(len(list(predicted_sl.keys())))
    areas_low = np.zeros(len(list(predicted_sl.keys())))
    for i in range(len(predicted_sl[list(predicted_sl.keys())[0]])-1):
        xA = current[i,0]
        yA = current[i,1]
        xD = current[i+1,0]
        yD = current[i+1,1]
        for j,date in enumerate(list(predicted_sl.keys())):
            # mean shoreline
            xB = predicted_sl[date][i,0]
            yB = predicted_sl[date][i,1]
            xC = predicted_sl[date][i+1,0]
            yC = predicted_sl[date][i+1,1]
            area = area_quadrangle(xA,yA,xB,yB,xC,yC,xD,yD)
            
            areas_mean[j] += area
            if is_mae:
                # low-estimated shoreline
                xB_l = predicted_low[date][i,0]
                yB_l = predicted_low[date][i,1]
                xC_l = predicted_low[date][i+1,0]
                yC_l = predicted_low[date][i+1,1]
                area_l = area_quadrangle(xA,yA,xB_l,yB_l,xC_l,yC_l,xD,yD)
                # high-estimated shoreline
                xB_h = predicted_hig[date][i,0]
                yB_h = predicted_hig[date][i,1]
                xC_h = predicted_hig[date][i+1,0]
                yC_h = predicted_hig[date][i+1,1]
                area_h = area_quadrangle(xA,yA,xB_h,yB_h,xC_h,yC_h,xD,yD)
                if np.isnan(area_h) and np.isnan(area_l):
                    areas_hig[j] += area
                    areas_low[j] += area
                elif np.isnan(area_l):
                    areas_low[j] += area
                    areas_hig[j] += area_h
                elif np.isnan(area_h):
                    areas_low[j] += area_l
                    areas_hig[j] += area
                else:
                    areas_low[j] += area_l
                    areas_hig[j] += area_h
            else:
                areas_hig[j] += area
                areas_low[j] += area      
            
    # construct the dictionary and estimate threatened population using density
    areas = dict([])
    pop = None
    if density != None:
        pop = dict([])
    for i,date in enumerate(list(predicted_sl.keys())):
        areas[date] = [areas_low[i]/10000,areas_mean[i]/10000,areas_hig[i]/10000]
        if density != None:
            pop[date] = np.array(areas[date])*density/100 # divide by 100 because the density is in inhab/km2 and the area is in ha
    
    # correct the interval if estimations are not in the right order
    for d in areas.keys():
        areas[d] = np.sort(areas[d])
        pop[d] = np.sort(pop[d])

    # plot estimated threatened population with a curve for each boundary
    if plot and (pop!=None):
        pop_low = []
        pop_mean = []
        pop_hig = []
        for k in pop.keys():
            pop_low.append(pop[k][0])
            pop_mean.append(pop[k][1])
            pop_hig.append(pop[k][2])
        
        fig = plt.figure(figsize=[15,8], tight_layout=True)
        if is_mae:
            plt.plot(list(predicted_sl.keys()),pop_low,'+-',c='green',label="Low estimation")
            plt.plot(list(predicted_sl.keys()),pop_mean,'+-',c='blue',label="Mean estimation")
            plt.plot(list(predicted_sl.keys()),pop_hig,'+-',c='red',label="High estimation")
            plt.fill_between(list(predicted_sl.keys()), pop_low, pop_hig, alpha=0.3)
        else:
            plt.plot(list(predicted_sl.keys()),pop_mean,'+-',c='blue')
        plt.grid()
        plt.legend(prop={'size': 10})
        plt.title('Estimated threatened population')
        plt.show()
    
    return areas, pop

