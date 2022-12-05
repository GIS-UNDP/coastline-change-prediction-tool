# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("..")
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()
import pandas as pd
from datetime import datetime
from python import extract_shorelines as es
from python import analyze_shoreline as asl
from python import correct_tides as ct
from python import predict as pt
from python import reconstruct_shoreline as rs
from python import estimate_pop as ep
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
os.chdir("../")

import pickle



# region of interest (longitude, latitude)
# polygon = [[[175.953076,  -2.475584],  'TAMANA'
#             [176.013947, -2.475584],   
#             [176.013947, -2.523176],
#             [175.953076, -2.523176],
#             [175.953076,  -2.475584]]]

# polygon = [[[151.301454, -33.700754], # NARRA
#             [151.311453, -33.702075],
#             [151.307237, -33.739761],
#             [151.294220, -33.736329],
#             [151.301454, -33.700754]]]

# polygon = [[[172.97097207630773,  1.3339524309132496], #'TARAWA'
#             [172.98858053913264, 1.3339524309132496],
#             [172.98858053913264, 1.3247216646721878],
#             [172.97097207630773, 1.3247216646721878],
#             [172.97097207630773,  1.3339524309132496]]]

# polygon = [[[-10.812718, 6.334993],  # West Point
#             [-10.802246, 6.334965], 
#             [-10.802246, 6.321754],
#             [-10.812718, 6.321754],
#             [-10.812718, 6.334993]]] 

# polygon = [[[34.31030273436481,31.40966528243077], #'Gaza'
#             [34.28535800052927,31.38761407527643],
#             [34.29494165934159,31.38285885930734],
#             [34.3196982003549,31.40502068816904],
#             [34.31030273436481,31.40966528243077]]]

polygon = [[[0.9840004042206285,5.868659802766709], #'Ghana'
            [1.002491492775628,5.868997660770736],
            [1.003566497639035,5.912882741342837],
            [0.9835488890625466,5.913424406492211],
            [0.9840004042206285,5.868659802766709]]]

# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
# polygon = SDS_tools.smallest_rectangle(polygon)
# # date range
# dates = ['2000-01-01', '2021-01-01']
# # satellite missions
# sat_list = ['S2']
# # name of the site
# sitename = 'TARAWA'
sitename = 'GHANA'
# sitename = 'GAZA1'
# sitename = 'WestPoint'
# region of interest (longitude, latitude)
# polygon = [[[-43.34948117649324,-23.00150013925698], #'Rio'
#             [-43.35045550564293,-23.01562998617737],
#             [-43.30821145398064,-23.01961931261296],
#             [-43.30540721465849,-23.00772275485573],
#             [-43.34948117649324,-23.00150013925698]]]

# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = SDS_tools.smallest_rectangle(polygon)
# date range
dates = ['2010-01-01', '2021-01-01']
# satellite missions
sat_list = ['S2']
# name of the site
# sitename = 'Rio'
# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'data')
# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}

# before downloading the images, check how many images are available for your inputs
#SDS_download.check_images_available(inputs);

# check if there is a 'references' dir containing predefined metadata
is_ref = os.path.isdir(os.path.join(filepath,sitename,'references'))
    
# redefine if files do not exist yet
redefine = not(os.path.exists(os.path.join(filepath,sitename,'%s_metadata.pkl'%(sitename))))
    
# actually we do not redefine the metadata if there are the references data
if (redefine and is_ref):
    redefine = False

# or the user can choose to crash existing files and redefine metadata
# or even force to continue with existing files (/!\ may lead to errors /!\)
# redefine = False ##### TO REMOVE ###########

print('redefine = ',redefine)


# with open(os.path.join(filepath,sitename,'mae_Holt.pkl'), 'rb') as f:
#     data1 = pickle.load(f)

# with open(os.path.join(filepath,sitename,'rmse_Holt.pkl'), 'rb') as k:
#     data2 = pickle.load(k)

if redefine:
    # this function retrieves the satellite images from Google Earth Engine 
    inputs['include_T2'] = True
    metadata = SDS_download.retrieve_images(inputs)
else:
    # if images already retrieved, just load the metadata file by only running the function below
    metadata = SDS_download.get_metadata(inputs)

settings = { 
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'output_epsg': 3857,        # epsg code of spatial reference system desired for the output   
    'pan_off': True, 
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) for buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 2300,      # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': 'default',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    # add the inputs defined previously
    'inputs': inputs
}

# options available only if the metadata and the files have to be redefined
if redefine:
    save_jpg = True    

# [optional] create a reference shoreline (helps to identify outliers and false detections)
ref_shoreline = True
if ref_shoreline:
    # %matplotlib qt
    settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
    # set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
    settings['max_dist_ref'] = 200  

if redefine:
    output = es.extract_shorelines(metadata, settings, inputs, plot=True)

if not(redefine):
    filepath = os.path.join(inputs['filepath'], sitename)
    with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
        output = pickle.load(f) 

if (redefine):
    # the user will interactively draw the shore-normal transects along the beach by calling:
    transects = SDS_transects.draw_transects(output, settings)
else:
    # the user will load the transect coordinates (make sure the spatial reference system is the same 
    # as defined previously by the parameter *output_epsg*) from a .geojson file by calling:
    #transects = SDS_transects.draw_transects(output, settings)
    geojson_file = os.path.join(inputs['filepath'],sitename, '%s_transects.geojson'%(sitename))
    transects = SDS_tools.transects_from_geojson(geojson_file)

# defines the along-shore distance over which to consider shoreline points to compute the median intersection (robust to outliers)
settings['along_dist'] = 25 
# compute and plot the time series
cross_distance = asl.analyze_shoreline(output,transects,settings,plot=True)

reference_elevation = 0.0 # elevation at which you would like the shoreline time-series to be
beach_slope = 0.020670

cross_distance = ct.correct_tides(cross_distance,settings,output,reference_elevation,beach_slope,plot=False)

corrected_sl = rs.reconstruct_shoreline(cross_distance,transects,output['dates'],output,inputs,settings,len(output['dates']),save_corrections=True)


# define the number of years to be predicted
n_years_further = 4

message, validity = pt.validate_year(n_years_further, output)

# model could be: Holt, ARIMA or SARIMAX
model = 'Holt'

compute_error = True
if compute_error:
    best_param, rmse, mae = pt.model_evaluation(cross_distance, n_years_further, metadata, output, settings, validity, model=model, smooth_data=False, smooth_coef=30, best_param=True, seasonality=2, MNDWI=False)
    print("Best parameters: ",best_param)
    print('======================================================')
    print("Mean absolute error: \n", mae)

time_series_pred, dates_pred_m = pt.predict(cross_distance,output,inputs,settings,n_years_further,validity,model=model,param=None,smooth_data=True,smooth_coef=1,seasonality=2,plot=True)

predicted_sl = rs.reconstruct_shoreline(time_series_pred,transects,dates_pred_m,output,inputs,settings,n_years_further)

areas, pop = ep.estimate_pop(cross_distance,predicted_sl,time_series_pred,dates_pred_m,n_years_further,transects,output,inputs,settings,density=0)
