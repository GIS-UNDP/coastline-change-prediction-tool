"""
This module contains all the functions needed to download the satellite images
from the Google Earth Engine server

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""


# load basic modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# earth engine module
import ee

# modules to download, unzip and stack the images
from urllib.request import urlretrieve
import zipfile
import copy
import shutil
from osgeo import gdal

# additional modules
from datetime import datetime, timedelta
import pytz
import pickle
from skimage import morphology, transform
from scipy import ndimage

# CoastSat modules
from coastsat import SDS_preprocess, SDS_tools, gdal_merge

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans
gdal.PushErrorHandler('CPLQuietErrorHandler')

# Main function to download images from the EarthEngine server
def retrieve_images(inputs):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2
    covering the area of interest and acquired between the specified dates.
    The downloaded images are in .TIF format and organised in subfolders, divided
    by satellite mission. The bands are also subdivided by pixel resolution.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    
    # initialise connection with GEE server
    ee.Initialize()

    # check image availabiliy and retrieve list of images
    im_dict_T1 = check_images_available(inputs)

     # create a new directory for this site with the name of the site
    im_folder = os.path.join(inputs['filepath'],inputs['sitename'])
    if not os.path.exists(im_folder): os.makedirs(im_folder)

    print('\nDownloading images:')
    suffix = '.tif'
    for satname in im_dict_T1.keys():
        print('%s: %d images'%(satname,len(im_dict_T1[satname])))
        # create subfolder structure to store the different bands
        filepaths = create_folder_structure(im_folder, satname)
        # initialise variables and loop through images
        georef_accs = []; filenames = []; all_names = []; im_epsg = []
        for i in range(len(im_dict_T1[satname])):

            im_meta = im_dict_T1[satname][i]

            # get time of acquisition (UNIX time) and convert to datetime
            t = im_meta['properties']['system:time_start']
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

            # get epsg code
            im_epsg.append(int(im_meta['bands'][0]['crs'][5:]))

            # get geometric accuracy
            if satname in ['L8']:
                if 'GEOMETRIC_RMSE_MODEL' in im_meta['properties'].keys():
                    acc_georef = im_meta['properties']['GEOMETRIC_RMSE_MODEL']
                else:
                    acc_georef = 12 # default value of accuracy (RMSE = 12m)
           
            georef_accs.append(acc_georef)

            bands = dict([])
            im_fn = dict([])
            # first delete dimensions key from dictionnary
            # otherwise the entire image is extracted (don't know why)
            im_bands = im_meta['bands']
            for j in range(len(im_bands)): del im_bands[j]['dimensions']

            # Landsat 8 download
            if satname in ['L8'] and inputs['bands'][0] == 4:
                # bands['pan'] = [im_bands[7]] # panchromatic band
                bands['ms'] = [im_bands[3]] 
            elif inputs['bands'][0]  == 5:
                bands['ms'] = [im_bands[4]] 

            elif inputs['bands'][0]  == 10:
                bands['ms'] = [im_bands[9]]

            else: 
                bands['ms'] = [im_bands[11]]
                                   
            for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + suffix
             
                # download .tif from EE (panchromatic band and multispectral bands)
            while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        # local_data_pan = download_tif(im_ee, inputs['polygon'], bands['pan'], filepaths[1])
                        local_data_ms = download_tif(im_ee, inputs['polygon'], bands['ms'], filepaths[2], inputs['sitename'], im_date)
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                #try: # panchromatic
                #     os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan']))
                # except: # overwrite if already exists
                #     os.remove(os.path.join(filepaths[1], im_fn['pan']))
                # #     os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan']))
                # try: # multispectral
                #     os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))
                # except: # overwrite if already exists
                #     os.remove(os.path.join(filepaths[2], im_fn['ms']))
                    os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))
                # metadata for .txt file
            # filename_txt = im_fn['pan'].replace('_pan','').replace('.tif','')
            # metadict = {'filename':im_fn['pan'],'acc_georef':georef_accs[i],
                            # 'epsg':im_epsg[i]}

            # write metadata
            # with open(os.path.join(filepaths[0],filename_txt + '.txt'), 'w') as f:
            #     for key in metadict.keys():
            #         f.write('%s\t%s\n'%(key,metadict[key]))
            # print percentage completion for user
            print('\r%d%%' %int((i+1)/len(im_dict_T1[satname])*100), end='')

        print('')

    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs)
    # merge overlapping images (necessary only if the polygon is at the boundary of an image)
    
    # save metadata dict
    with open(os.path.join(im_folder, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return metadata

# function to load the metadata if images have already been downloaded
def get_metadata(inputs):
    """
    Gets the metadata from the downloaded images by parsing .txt files located
    in the \meta subfolder.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # directory containing the images
    filepath = os.path.join(inputs['filepath'],inputs['sitename'])
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions
    for satname in ['L5','L7','L8','L9','S2']:
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(filepath, satname, 'meta')
            # get the list of filenames and sort it chronologically
            filenames_meta = os.listdir(filepath_meta)
            filenames_meta.sort()
            # loop through the .txt files
            for im_meta in filenames_meta:
                # read them and extract the metadata info: filename, georeferencing accuracy
                # epsg code and date
                with open(os.path.join(filepath_meta, im_meta), 'r') as f:
                    filename = f.readline().split('\t')[1].replace('\n','')
                    acc_georef = float(f.readline().split('\t')[1].replace('\n',''))
                    epsg = int(f.readline().split('\t')[1].replace('\n',''))
                date_str = filename[0:19]
                date = pytz.utc.localize(datetime(int(date_str[:4]),int(date_str[5:7]),
                                                  int(date_str[8:10]),int(date_str[11:13]),
                                                  int(date_str[14:16]),int(date_str[17:19])))
                # store the information in the metadata dict
                metadata[satname]['filenames'].append(filename)
                metadata[satname]['acc_georef'].append(acc_georef)
                metadata[satname]['epsg'].append(epsg)
                metadata[satname]['dates'].append(date)

    # save a .pkl file containing the metadata dict
    with open(os.path.join(filepath, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return metadata


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################

def check_images_available(inputs):
    """
    Scan the GEE collections to see how many images are available for each
    satellite mission (L5,L7,L8,L9,S2), collection (C01,C02) and tier (T1,T2).

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict
        inputs dictionnary

    Returns:
    -----------
    im_dict_T1: list of dict
        list of images in Tier 1 and Level-1C
    im_dict_T2: list of dict
        list of images in Tier 2 (Landsat only)
    """

    dates = [datetime.strptime(_,'%Y-%m-%d') for _ in inputs['dates']]
    dates_str = inputs['dates']
    polygon = inputs['polygon']
    
    # check if dates are in chronological order
    if  dates[1] <= dates[0]:
        raise Exception('Verify that your dates are in the correct chronological order')

    # check if EE was initialised or not
    try:
        ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')
    except:
        ee.Initialize()
        
    print('Number of images available between %s and %s:'%(dates_str[0],dates_str[1]), end='\n')
    
    # get images in Landsat Tier 1 as well as Sentinel Level-1C
    print('- In Landsat Tier 1 & Sentinel-2 Level-1C:')
    col_names_T1 = {'L5':'LANDSAT/LT05/C01/T1_TOA', #%inputs['landsat_collection'],
                    'L7':'LANDSAT/LE07/C01/T1_TOA', #%inputs['landsat_collection'],
                    'L8':'LANDSAT/LC08/C02/T1_TOA', #%inputs['landsat_collection'],
                    'L9':'LANDSAT/LC09/C02/T1_TOA', # only C02 for Landsat 9
                    'S2':'COPERNICUS/S2'}
    im_dict_T1 = dict([])
    sum_img = 0
    for satname in inputs['sat_list']:
        im_list = get_image_info(col_names_T1[satname],satname,polygon,dates_str)
        sum_img = sum_img + len(im_list)
        print('     %s: %d images'%(satname,len(im_list)))
        im_dict_T1[satname] = im_list
            
        
    print('  Total to download: %d images'%sum_img)

    # if only S2 is in sat_list, stop here as no Tier 2 for Sentinel
    if len(inputs['sat_list']) == 1 and inputs['sat_list'][0] == 'S2':
        return im_dict_T1, []

  
    return im_dict_T1


def get_image_info(collection,satname,polygon,dates):
    """
    Reads info about EE images for the specified collection, satellite and dates

    KV WRL 2022

    Arguments:
    -----------
    collection: str
        name of the collection (e.g. 'LANDSAT/LC08/C02/T1_TOA')
    satname: str
        name of the satellite mission
    polygon: list
        coordinates of the polygon in lat/lon
    dates: list of str
        start and end dates (e.g. '2022-01-01')

    Returns:
    -----------
    im_list: list of ee.Image objects
        list with the info for the images
    """
    while True:
        try:
            # get info about images
            ee_col = ee.ImageCollection(collection)
            col = ee_col.filterBounds(ee.Geometry.Polygon(polygon))\
                        .filterDate(dates[0],dates[1])
            im_list = col.getInfo().get('features')
            break
        except:
            continue
    # remove very cloudy images (>95% cloud cover)
    im_list = remove_cloudy_images(im_list, satname)
    return im_list


def download_tif(image, polygon, bandsId, filepath, sitename, dates):
    """
    Downloads a .TIF image from the ee server. The image is downloaded as a
    zip file then moved to the working directory, unzipped and stacked into a
    single .TIF file.

    Two different codes based on which version of the earth-engine-api is being
    used.

    KV WRL 2018

    Arguments:
    -----------
    image: ee.Image
        Image object to be downloaded
    polygon: list
        polygon containing the lon/lat coordinates to be extracted
        longitudes in the first column and latitudes in the second column
    bandsId: list of dict
        list of bands to be downloaded
    filepath: location where the temporary file should be saved

    Returns:
    -----------
    Downloads an image in a file named data.tif

    """

    # for the old version of ee only
    if int(ee.__version__[-3:]) <= 201:
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image.serialize(),
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': 'data',
            }))
        local_zip, headers = urlretrieve(url)
        with zipfile.ZipFile(local_zip) as local_zipfile:
            return local_zipfile.extract('data.tif', filepath)
    # for the newer versions of ee
    else:
        # crop image on the server and create url to download
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image,
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': sitename + '_' + dates,
            }))
        # download zipfile with the cropped bands
        local_zip, headers = urlretrieve(url)
        # move zipfile from temp folder to data folder
        dest_file = os.path.join(filepath, 'imagezip')
        shutil.move(local_zip,dest_file)
        # unzip file
        with zipfile.ZipFile(dest_file) as local_zipfile:
            for fn in local_zipfile.namelist():
                local_zipfile.extract(fn, filepath)
            # filepath + filename to single bands
            fn_tifs = [os.path.join(filepath,_) for _ in local_zipfile.namelist()]
        # stack bands into single .tif
        # outds = gdal.BuildVRT(os.path.join(filepath,'stacked.vrt'), fn_tifs, separate=True)
        # outds = gdal.Translate(os.path.join(filepath,'data.tif'), outds)
        # # delete single-band files
        # for fn in fn_tifs: os.remove(fn)
        # delete .vrt file
        # os.remove(os.path.join(filepath,'stacked.vrt'))
        # delete zipfile
        os.remove(dest_file)
        # delete data.tif.aux (not sure why this is created)
        if os.path.exists(os.path.join(filepath,'data.tif.aux')):
            os.remove(os.path.join(filepath,'data.tif.aux'))
        # return filepath to stacked file called data.tif
        # return os.path.join(filepath,'data.tif')


def create_folder_structure(im_folder, satname):
    """
    Create the structure of subfolders for each satellite mission

    KV WRL 2018

    Arguments:
    -----------
    im_folder: str
        folder where the images are to be downloaded
    satname:
        name of the satellite mission

    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created
    """

    # one folder for the metadata (common to all satellites)
    filepaths = [os.path.join(im_folder, satname, 'meta')]
    # subfolders depending on satellite mission
    if satname == 'L5':
        filepaths.append(os.path.join(im_folder, satname, '30m'))
    elif satname in ['L7','L8','L9']:
        filepaths.append(os.path.join(im_folder, satname, 'pan'))
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
    elif satname in ['S2']:
        filepaths.append(os.path.join(im_folder, satname, '10m'))
        filepaths.append(os.path.join(im_folder, satname, '20m'))
        filepaths.append(os.path.join(im_folder, satname, '60m'))
    # create the subfolders if they don't exist already
    for fp in filepaths:
        if not os.path.exists(fp): os.makedirs(fp)

    return filepaths


def remove_cloudy_images(im_list, satname, prc_cloud_cover=95):
    """
    Removes from the EE collection very cloudy images (>95% cloud cover)

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection
    satname:
        name of the satellite mission
    prc_cloud_cover: int
        percentage of cloud cover acceptable on the images

    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """

    # remove very cloudy images from the collection (>95% cloud)
    if satname in ['L5','L7','L8','L9']:
        cloud_property = 'CLOUD_COVER'
    elif satname in ['S2']:
        cloud_property = 'CLOUDY_PIXEL_PERCENTAGE'
    cloud_cover = [_['properties'][cloud_property] for _ in im_list]
    if np.any([_ > prc_cloud_cover for _ in cloud_cover]):
        idx_delete = np.where([_ > prc_cloud_cover for _ in cloud_cover])[0]
        im_list_upt = [x for k,x in enumerate(im_list) if k not in idx_delete]
    else:
        im_list_upt = im_list

    return im_list_upt

