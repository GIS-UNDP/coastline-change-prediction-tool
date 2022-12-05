#%% load modules
import os
import matplotlib
import matplotlib.pyplot as plt
from coastsat import SDS_preprocess, SDS_shoreline, SDS_tools
#%%

# Batch shoreline detection
def extract_shorelines(metadata,settings,inputs,save_jpg=False,plot=True,geomtype='lines'):
    # save .jpg files of the preprocessed satellite images
    if (save_jpg):
        SDS_preprocess.save_jpg(metadata, settings)
    
    # extract shorelines from all images (also saves output.pkl and shorelines.kml)
    output = SDS_shoreline.extract_shorelines(metadata, settings)
    
    # remove duplicates (images taken on the same date by the same satellite)
    output = SDS_tools.remove_duplicates(output)
    # remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)
    
    # for GIS applications, save output into a GEOJSON layer
    gdf = SDS_tools.output_to_gdf(output, geomtype)
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
    # save GEOJSON layer to file
    gdf.to_file(os.path.join(inputs['filepath'], inputs['sitename'], '%s_output_%s.geojson'%(inputs['sitename'],geomtype)),
                                    driver='GeoJSON', encoding='utf-8')
#    # save SHP layer to file (in dedicated folder)
#    filepath_shp = os.path.join(inputs['filepath'], inputs['sitename'], 'output_%s'%(geomtype)+'_shapefiles')
#    if not(os.path.exists(filepath_shp)):
#        os.makedirs(filepath_shp,exist_ok=False) 
#    for i in range(len(output['dates'])):
#        path_shp_file = os.path.join(filepath_shp,'%s_output_%s_%s'%(inputs['sitename'],geomtype,output['dates'][i].date().strftime('%Y-%m-#%d')))
#        if not(os.path.exists(path_shp_file)):
#            os.makedirs(path_shp_file,exist_ok=False) 
#        path_shp_file = os.path.join(path_shp_file,'%s_output_%s_%s'%(inputs['sitename'],geomtype,output['dates'][i].date().strftime('%Y-#%m-%d')+'.shp'))
#        gdf_sep = SDS_tools.output_to_gdf(output, geomtype)
#        gdf_sep.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
#        gdf_sep.to_file(path_shp_file, driver='ESRI Shapefile', encoding='utf-8')
        
    # save SHP layer to file (in dedicated folder)
    filepath_shp = os.path.join(inputs['filepath'], inputs['sitename'], 'output_%s'%(geomtype)+'_shapefiles')
    if not(os.path.exists(filepath_shp)):
        os.makedirs(filepath_shp,exist_ok=False) 
    for i in range(len(output['dates'])):
        path_shp_file = os.path.join(filepath_shp,'%s_output_%s_%s'%(inputs['sitename'],geomtype,output['dates'][i].date().strftime('%Y-%m-%d')))
        if not(os.path.exists(path_shp_file)):
            os.makedirs(path_shp_file,exist_ok=False)
        path_shp_file = os.path.join(path_shp_file,'%s_output_%s_%s'%(inputs['sitename'],geomtype,output['dates'][i].date().strftime('%Y-%m-%d')+'.shp'))
        tmp = output.copy()
        tmp['dates'] = [output['dates'][i]]
        tmp['shorelines'] = [output['shorelines'][i]]
        tmp['filename'] = [output['filename'][i]]
        gdf_pred = SDS_tools.output_to_gdf(tmp, geomtype)
        gdf_pred.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
        gdf_pred.to_file(path_shp_file, driver='ESRI Shapefile', encoding='utf-8')
    
    #if not(os.path.isdir(os.path.join(inputs['filepath'], inputs['sitename'], 'output_%s_shapefiles'%(geomtype)))):
    #    os.makedirs(os.path.join(inputs['filepath'], inputs['sitename'], 'output_%s_shapefiles'%(geomtype))) 
    #gdf.to_file(os.path.join(inputs['filepath'], inputs['sitename'],'output_%s_shapefiles'%(geomtype),
    #                         '%s_output_%s.shp'%(inputs['sitename'],geomtype)), driver='ESRI Shapefile', encoding='utf-8')
        
    if plot:
        # plot the mapped shorelines
        fig = plt.figure(figsize=[15,8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            color=matplotlib.cm.jet(i/len(output['shorelines']))
            plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'), markersize=2, color=color)
        plt.title('Mapped shorelines',pad=-10)
        plt.legend(ncol=3, prop={'size': 6}) 
    
    return output