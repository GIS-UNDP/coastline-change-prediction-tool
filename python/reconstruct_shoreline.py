# %% load modules
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import skimage.transform as transform
from coastsat import SDS_tools

# %% SUB-FUNCTIONS
# keep the results for each year of the prediction dates range
def keep_coords(coords,dates_pred,output,save_corrections):
    new_coords=dict([])
    for n in range(len(dates_pred)):
        if dates_pred[n].date()>output['dates'][-1].date() and dates_pred[n].date().month==output['dates'][-1].date().month and not(save_corrections):
            tmp=[]
            for k in coords.keys():
                tmp.append(coords[k][n])
            new_coords[dates_pred[n].date().strftime('%Y-%m')]=np.array(tmp)
        if save_corrections:
            tmp=[]
            for k in coords.keys():
                if n<len(coords[k]):
                    tmp.append(coords[k][n])
            new_coords[dates_pred[n].date().strftime('%Y-%m')]=np.array(tmp)
    return new_coords
  
# interpolate between predicted points
def interpolate(coords):
    coords_interp=dict([])
    for i in coords.keys():
        interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
        for k in range(len(coords[i])-1):
            pt_dist = np.linalg.norm(coords[i][k,:]-coords[i][k+1,:])
            if np.isnan(pt_dist):
                continue
            xvals = np.arange(0,pt_dist)
            yvals = np.zeros(len(xvals))
            pt_coords = np.zeros((len(xvals),2))
            pt_coords[:,0] = xvals
            pt_coords[:,1] = yvals
            phi = 0
            deltax = coords[i][k+1,0] - coords[i][k,0]
            deltay = coords[i][k+1,1] - coords[i][k,1]
            phi = np.pi/2 - np.math.atan2(deltax, deltay)
            tf = transform.EuclideanTransform(rotation=phi, translation=coords[i][k,:])
            interp = np.append(interp,tf(pt_coords), axis=0)
        interp = np.delete(interp,0,axis=0)
        coords_interp[i]=interp
    return coords_interp

# save the predicted shorelines in a GeoJSON file and shapefiles
def save_prediction(coords_interp,dates_pred,output,inputs,settings,geomtype,corrected_sl=False):
    if corrected_sl:
        add_title = '_tidally_corrected'
    else:
        add_title = '_prediction'
    output_pred=output.copy()
    for k in output_pred.keys():
        output_pred[k]=len(output[k])*[0]
    output_pred['dates'] = []
    output_pred['shorelines'] = []
    output_pred['filename'] = []
    for i in range(len(list(coords_interp.keys()))):
        output_pred['dates'].append(datetime.datetime.strptime(list(coords_interp.keys())[i], '%Y-%m'))
        output_pred['shorelines'].append(coords_interp[list(coords_interp.keys())[i]])
        output_pred['filename'].append(list(coords_interp.keys())[i]+'_S2_WestPoint_10m_dup'+add_title+'.tif')
    gdf_pred = SDS_tools.output_to_gdf(output_pred, geomtype)
    gdf_pred.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
    # save GEOJSON layer to file
    gdf_pred.to_file(os.path.join(inputs['filepath'], inputs['sitename'], '%s_output_%s'%(inputs['sitename'],geomtype)+add_title+'.geojson'),
                                    driver='GeoJSON', encoding='utf-8')
    # save SHP layer to file (in dedicated folder)
    filepath_shp = os.path.join(inputs['filepath'], inputs['sitename'], 'output_%s'%(geomtype)+add_title+'_shapefiles')
    if not os.path.exists(filepath_shp):
        os.makedirs(filepath_shp,exist_ok=False) 
    for i in range(len(output_pred['dates'])):
        path_shp_file = os.path.join(filepath_shp,'%s_output_%s_%s'%(inputs['sitename'],geomtype,list(coords_interp.keys())[i])+add_title)
        if not os.path.exists(path_shp_file):
            os.makedirs(path_shp_file,exist_ok=False) 
        path_shp_file = os.path.join(path_shp_file,'%s_output_%s_%s'%(inputs['sitename'],geomtype,list(coords_interp.keys())[i])+add_title+'.shp')
        tmp = output_pred.copy()
        tmp['dates'] = [output_pred['dates'][i]]
        tmp['shorelines'] = [output_pred['shorelines'][i]]
        tmp['filename'] = [output_pred['filename'][i]]
        gdf_pred = SDS_tools.output_to_gdf(tmp, geomtype)
        gdf_pred.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
        gdf_pred.to_file(path_shp_file, driver='ESRI Shapefile', encoding='utf-8')
    return
# %%

# get the coordinates of the predicted shoreline from the predicted lengths of the transects
def reconstruct_shoreline(time_series,transects,dates_pred,output,inputs,settings,n_steps_further, estimate_pop=False,save_corrections=False,geomtype='lines'):
    n_steps_further = n_steps_further * 12
    if estimate_pop:
        plot = False
        save_geo = False
    elif save_corrections:
        plot = False
        save_geo = True
    else:
        plot = True
        save_geo = True
    
    # get the coordinates of the different points from the predicted distances along each transect
    coords_pts_pred = dict([])
    for key in transects.keys():
        l=abs(transects[key][1,0]-transects[key][0,0])
        h=abs(transects[key][1,1]-transects[key][0,1])
        tan=h/l
        tmp=[]
        d=[]
        for n in range(len(time_series[key])):
            if (dates_pred[n].date()>=output['dates'][-1].date() and not(save_corrections)) or save_corrections:
                c=np.sqrt(time_series[key][n]**2/(tan**2+1))
                r=c*tan
                if l!=0.0:
                    if transects[key][0,1]>transects[key][1,1]: 
                        r=transects[key][0,1]-r
                    else:
                        r=transects[key][0,1]+r     
                if h!=0.0:
                    if transects[key][1,0]>transects[key][0,0]: 
                        c=transects[key][0,0]+c
                    else:
                        c=transects[key][0,0]-c
                if l==0.0:
                    if transects[key][0,1]>transects[key][1,1]: 
                        r=transects[key][0,1]-time_series[key][n]
                    else:
                        r=transects[key][0,1]+time_series[key][n]
                    c=transects[key][0,0]
                if h==0.0:
                    if transects[key][1,0]>transects[key][0,0]: 
                        c=transects[key][0,0]+time_series[key][n]
                    else:
                        c=transects[key][0,0]-time_series[key][n]
                    r=transects[key][0,1]
                tmp.append([c,r])
                d.append(dates_pred[n])
        dates=d
        coords_pts_pred[key]=tmp
    
    # keep one prediction per year
    if not(estimate_pop) or (estimate_pop and len(coords_pts_pred['1'])>1):
        coords_pts_pred = keep_coords(coords_pts_pred,dates,output,save_corrections)
        
    # interpolate to construct a smoother shoreline
    coords_interp = interpolate(coords_pts_pred)
   
    if plot:
        # plot the predicted shorelines
        fig = plt.figure(figsize=[15,8])
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i,date in enumerate(list(coords_interp.keys())):
            x = coords_interp[date][:,0]
            y = coords_interp[date][:,1]
            ind = np.argsort(y)
            x = x[ind]
            y = np.sort(y)
            color=matplotlib.cm.jet(i/len(list(coords_interp.keys())))
            plt.plot(x, y, '.', label=date, markersize=1, color=color)
        plt.legend(ncol=3, prop={'size': 6});

        # plot and compare with the previous shorelines
        fig = plt.figure(figsize=[15,8])
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            color=matplotlib.cm.jet(i/(len(output['shorelines'])+len(coords_interp)+1))
            plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%Y-%m'), markersize=2, color=color)
        for i,date in enumerate(list(coords_interp.keys())):
            x = coords_interp[date][:,0]
            y = coords_interp[date][:,1]
            ind = np.argsort(y)
            x = x[ind]
            y = np.sort(y)
            color=matplotlib.cm.jet((i+len(output['shorelines']))/(len(coords_interp)+1+len(output['shorelines'])))
            plt.plot(x, y, '.', label=date+' Prediction', markersize=2, color=color)
        plt.title('Mapped shorelines and predictions')
        plt.legend(ncol=3, prop={'size': 6});
    
    # save the predictions in GeoJSON format and shapefiles
    if save_geo:
        save_prediction(coords_interp,dates_pred,output,inputs,settings,geomtype,save_corrections)
    return coords_pts_pred
