# CoastSat

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2779293.svg)](https://doi.org/10.5281/zenodo.2779293)
[![Join the chat at https://gitter.im/CoastSat/community](https://badges.gitter.im/spyder-ide/spyder.svg)](https://gitter.im/CoastSat/community)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/github/release/kvos/CoastSat)](https://GitHub.com/kvos/CoastSat/releases/)

CoastSat is an open-source software toolkit written in Python that enables users to obtain time-series of shoreline position at any coastline worldwide from 30+ years (and growing) of publicly available satellite imagery.

![Alt text](https://github.com/kvos/CoastSat/blob/master/doc/example.gif)

:point_right: Relevant publications:

- Shoreline detection algorithm: https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
- Accuracy assessment and applications: https://doi.org/10.1016/j.coastaleng.2019.04.004
- Beach slope estimation: https://doi.org/10.1029/2020GL088365 (preprint [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2))
- Satellite-derived shorelines along meso-macrotidal beaches: https://doi.org/10.1016/j.geomorph.2021.107707
- Beach-face slope dataset for Australia: https://doi.org/10.5194/essd-14-1345-2022

:point_right: Other repositories and addons related to this toolbox:
- [CoastSat.slope](https://github.com/kvos/CoastSat.slope): estimates the beach-face slope from the satellite-derived shorelines obtained with CoastSat.
- [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope): shoreline extraction for PlanetScope Dove imagery (near-daily since 2017 at 3m resolution).
- [InletTracker](https://github.com/VHeimhuber/InletTracker): monitoring of intermittent open/close estuary entrances.
- [CoastSat.islands](https://github.com/mcuttler/CoastSat.islands): 2D planform measurements for small reef islands.
- [CoastSeg](https://github.com/dbuscombe-usgs/CoastSeg): image segmentation, deep learning, doodler.
- [CoastSat.Maxar](https://github.com/kvos/CoastSat.Maxar): shoreline extraction on Maxar World-View images (in progress)

:point_right: Visit the [CoastSat website](http://coastsat.wrl.unsw.edu.au/) to explore and download regional-scale datasets of satellite-derived shorelines and beach slopes generated with CoastSat.

:star: **If you like the repo put a star on it!** :star:

### Latest updates
:arrow_forward: *(2022/07/20)*
Option to disable panchromatic sharpening on Landsat 7, 8 and 9 imagery. This setting is recommended for the time being as a bug has been reported with occasional misalignment between the panchromatic and multispectral bands downloaded from Google Earth Engine.

:arrow_forward: *(2022/05/02)*
Compatibility with Landsat 9 and Landsat Collection 2

### Project description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ field measurements are available. CoastSat enables the non-expert user to extract shorelines from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2 images.
The shoreline detection algorithm implemented in CoastSat is optimised for sandy beach coastlines. It combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

The toolbox has four main functionalities:
1. assisted retrieval from Google Earth Engine of all available satellite images spanning the user-defined region of interest and time period
2. automated extraction of shorelines from all the selected images using a sub-pixel resolution technique
3. intersection of the 2D shorelines with user-defined shore-normal transects
4. tidal correction using measured water levels and an estimate of the beach slope

## 1. Installation

### 1.1 Create an environment with Anaconda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have it installed on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to go the folder where you have downloaded this repository.

Create a new environment named `coastsat` with all the required packages by entering these commands in succession:

```
conda create -n coastsat python=3.8
conda activate coastsat
conda install spyder notebook
conda install gdal geopandas
conda install scikit-image

conda install -c conda-forge earthengine-api
conda install -c conda-forge astropy
conda install -c conda-forge opt_einsum
conda install -c conda-forge gast
conda install -c conda-forge statsmodels 
conda install -c conda-forge astunparse
conda install -c conda-forge termcolor
conda install -c conda-forge flatbuffers
conda install -c conda-forge python-flatbuffers
conda install -c conda-forge tensorflow keras

pip install pmdarima

```

All the required packages have now been installed in an environment called `coastsat`. Always make sure that the environment is activated with:

```
conda activate coastsat
```

To confirm that you have successfully activated CoastSat, your terminal command line prompt should now start with (coastsat).

:warning: **In case errors are raised** :warning:: clean things up with the following command (better to have the Anaconda Prompt open as administrator) before attempting to install `coastsat` again:
```
conda clean --all
```

You can also install the packages with the **Anaconda Navigator**, in the *Environments* tab. For more details, the following [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) shows how to create and manage an environment with Anaconda.

### 1.2 Activate Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

Once your request has been approved, with the `coastsat` environment activated, run the following command on the Anaconda Prompt to link your environment to the GEE server:

```
earthengine authenticate
```

A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal. If this steps is not working you need to install Google cloud. Find installation instructions here: https://cloud.google.com/sdk/docs/install. 

Now you are ready to start using the CoastSat toolbox!

**Note**: remember to always activate the environment with `conda activate coastsat` each time you are preparing to use the toolbox.

## 2. Usage

An example of how to run the software in a Jupyter Notebook is provided in the repository (`example_jupyter.ipynb`). To run this, first activate your `coastsat` environment with `conda activate coastsat` (if not already active), and then type:

```
jupyter notebook
```

A web browser window will open. Point to the directory where you downloaded this repository and click on `example_jupyter.ipynb`.

The following sections guide the reader through the different functionalities of CoastSat with an example at Narrabeen-Collaroy beach (Australia). If you prefer to use **Spyder**, **PyCharm** or other integrated development environments (IDEs), a Python script named `example.py` is also included in the repository.

If using `example.py` on **Spyder**, make sure that the Graphics Backend is set to **Automatic** and not **Inline** (as this mode doesn't allow to interact with the figures). To change this setting go under Preferences>IPython console>Graphics.

A Jupyter Notebook combines formatted text and code. To run the code, place your cursor inside one of the code sections and click on the `run cell` button (or press `Shift` + `Enter`) and progress forward.

![image](https://user-images.githubusercontent.com/7217258/165960239-e8870f7e-0dab-416e-bbdd-089b136b7d20.png)


### 2.1 Retrieval of the satellite images

To retrieve from the GEE server the available satellite images cropped around the user-defined region of coastline for the particular time period of interest, the following variables are required:
- `polygon`: the coordinates of the region of interest (longitude/latitude pairs in WGS84)
- `dates`: dates over which the images will be retrieved (e.g., `dates = ['2017-12-01', '2018-01-01']`)
- `sat_list`: satellite missions to consider (e.g., `sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']` for Landsat 5, 7, 8, 9 and Sentinel-2 collections)
- `sitename`: name of the site (this is the name of the subfolder where the images and other accompanying files will be stored)
- `filepath`: filepath to the directory where the data will be stored
- :new: `landsat_collection`: whether to use Collection 1 (`C01`) or Collection 2 (`C02`). Note that after 2022/01/01, Landsat images are only available in Collection 2. Landsat 9 is therefore only available as Collection 2. So if the user has selected `C01`, images prior to 2022/01/01 will be downloaded from Collection 1, while images captured after that date will be automatically taken from `C02`. Also note that at the time of writing `C02` is not complete in Google Earth Engine and still being uploaded.

The call `metadata = SDS_download.retrieve_images(inputs)` will launch the retrieval of the images and store them as .TIF files (under */filepath/sitename*). The metadata contains the exact time of acquisition (in UTC time) of each image, its projection and its geometric accuracy. If the images have already been downloaded previously and the user only wants to run the shoreline detection, the metadata can be loaded directly by running `metadata = SDS_download.get_metadata(inputs)`.

The screenshot below shows an example of inputs that will retrieve all the images of Collaroy-Narrabeen (Australia) acquired by Sentinel-2 in December 2017.

![doc1](https://user-images.githubusercontent.com/7217258/166197244-9f41de17-f387-40a6-945e-8a78b581c4b1.png)

**Note:** The area of the polygon should not exceed 100 km2, so for very long beaches split it into multiple smaller polygons.

### 2.2 Shoreline detection

To map the shorelines, the following user-defined settings are needed:
- `cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1 - this may require some initial experimentation).
- `output_epsg`: epsg code defining the spatial reference system of the shoreline coordinates. It has to be a cartesian coordinate system (i.e. projected) and not a geographical coordinate system (in latitude and longitude angles). See http://spatialreference.org/ to find the EPSG number corresponding to your local coordinate system. If unsure, use 3857 which is the web-mercator.
- `check_detection`: if set to `True` the user can quality control each shoreline detection interactively (recommended when mapping shorelines for the first time) and accept/reject each shoreline.
- `adjust_detection`: in case users wants more control over the detected shorelines, they can set this parameter to `True`, then they will be able to manually adjust the threshold used to map the shoreline on each image.
- `save_figure`: if set to `True` a figure of each mapped shoreline is saved under */filepath/sitename/jpg_files/detection*, even if the two previous parameters are set to `False`. Note that this may slow down the process.
- `pan_off`: set to `True` to disable panchromatic sharpening of Landsat 7, 8, 9 images. Down-sampled images to 15 m are used instead.

There are additional parameters (`min_beach_size`, `buffer_size`, `min_length_sl`, `cloud_mask_issue` and `sand_color`) that can be tuned to optimise the shoreline detection (for Advanced users only). For the moment leave these parameters set to their default values, we will see later how they can be modified.

An example of settings is provided here:

![Capture](https://user-images.githubusercontent.com/7217258/179879844-82f8ebea-0278-490f-9c79-111e5e363160.JPG)

Once all the settings have been defined, the batch shoreline detection can be launched by calling:
```
output = SDS_shoreline.extract_shorelines(metadata, settings)
```
When `check_detection` is set to `True`, a figure like the one below appears and asks the user to manually accept/reject each detection by pressing **on the keyboard** the `right arrow` (⇨) to `keep` the shoreline or `left arrow` (⇦) to `skip` the mapped shoreline. The user can break the loop at any time by pressing `escape` (nothing will be saved though).

![map_shorelines](https://user-images.githubusercontent.com/7217258/60766769-fafda480-a0f1-11e9-8f91-419d848ff98d.gif)

When `adjust_detection` is set to `True`, a figure like the one below appears and the user can adjust the position of the shoreline by clicking on the histogram of MNDWI pixel intensities. Once the threshold has been adjusted, press `Enter` and then accept/reject the image with the keyboard arrows.

![Alt text](https://github.com/kvos/CoastSat/blob/master/doc/adjust_shorelines.gif)

Once all the shorelines have been mapped, the output is available in two different formats (saved under */filepath/data/sitename*):
- `sitename_output.pkl`: contains a list with the shoreline coordinates, the exact timestamp at which the image was captured (UTC time), the geometric accuracy and the cloud cover of each individual image. This list can be manipulated with Python, a snippet of code to plot the results is provided in the example script.
- `sitename_output.geojson`: this output can be visualised in a GIS software (e.g., QGIS, ArcGIS).

The figure below shows how the satellite-derived shorelines can be opened in a GIS software (QGIS) using the `.geojson` output. Note that the coordinates in the `.geojson` file are in the spatial reference system defined by the `output_epsg`.

<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/7217258/49361401-15bd0480-f730-11e8-88a8-a127f87ca64a.jpeg">
</p>

#### Reference shoreline

Before running the batch shoreline detection, there is the option to manually digitize a reference shoreline on one cloud-free image. This reference shoreline helps to reject outliers and false detections when mapping shorelines as it only considers as valid shorelines the points that are within a defined distance from this reference shoreline.

 The user can manually digitize one or several reference shorelines on one of the images by calling:
```
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_manual(metadata, settings)
settings['max_dist_ref'] = 100 # max distance (in meters) allowed from the reference shoreline
```
This function allows the user to click points along the shoreline on cloud-free satellite images, as shown in the animation below.

![ref_shoreline](https://user-images.githubusercontent.com/7217258/70408922-063c6e00-1a9e-11ea-8775-fc62e9855774.gif)

The maximum distance (in metres) allowed from the reference shoreline is defined by the parameter `max_dist_ref`. This parameter is set to a default value of 100 m. If you think that 100 m buffer from the reference shoreline will not capture the shoreline variability at your site, increase the value of this parameter. This may be the case for large nourishments or eroding/accreting coastlines.

#### Advanced shoreline detection parameters

As mentioned above, there are some additional parameters that can be modified to optimise the shoreline detection:
- `min_beach_area`: minimum allowable object area (in metres^2) for the class 'sand'. During the image classification, some features (for example, building roofs) may be incorrectly labelled as sand. To correct this, all the objects classified as sand containing less than a certain number of connected pixels are removed from the sand class. The default value is 4500 m^2, which corresponds to 20 connected pixels of 15 m^2. If you are looking at a very small beach (<20 connected pixels on the images), try decreasing the value of this parameter.
- `buffer_size`: radius (in metres) that defines the buffer around sandy pixels that is considered to calculate the sand/water threshold. The default value of `buffer_size` is 150 m. This parameter should be increased if you have a very wide (>150 m) surf zone or inter-tidal zone.
- `min_length_sl`: minimum length (in metres) of shoreline perimeter to be valid. This can be used to discard small features that are detected but do not correspond to the actual shoreline. The default value is 200 m. If the shoreline that you are trying to map is shorter than 200 m, decrease the value of this parameter.
- `cloud_mask_issue`: the cloud mask algorithm applied to Landsat images by USGS, namely CFMASK, does have difficulties sometimes with very bright features such as beaches or white-water in the ocean. This may result in pixels corresponding to a beach being identified as clouds and appear as masked pixels on your images. If this issue seems to be present in a large proportion of images from your local beach, you can switch this parameter to `True` and CoastSat will remove from the cloud mask the pixels that form very thin linear features, as often these are beaches and not clouds. Only activate this parameter if you observe this very specific cloud mask issue, otherwise leave to the default value of `False`.
- `sand_color`: this parameter can take 3 values: `default`, `dark` or `bright`. Only change this parameter if you are seing that with the `default` the sand pixels are not being classified as sand (in orange). If your beach has dark sand (grey/black sand beaches), you can set this parameter to `dark` and the classifier will be able to pick up the dark sand. On the other hand, if your beach has white sand and the `default` classifier is not picking it up, switch this parameter to `bright`. At this stage this option is only available for Landsat images (soon for Sentinel-2 as well).

#### Re-training the classifier
CoastSat's shoreline mapping alogorithm uses an image classification scheme to label each pixel into 4 classes: sand, water, white-water and other land features. While this classifier has been trained using a wide range of different beaches, it may be that it does not perform very well at specific sites that it has never seen before. You can train a new classifier with site-specific training data in a few minutes by following the Jupyter notebook in [re-train CoastSat classifier](https://github.com/kvos/CoastSat/blob/master/doc/train_new_classifier.md).

### 2.3 Shoreline change analysis

This section shows how to obtain time-series of shoreline change along shore-normal transects. Each transect is defined by two points, its origin and a second point that defines its length and orientation. The origin is always defined first and located landwards, the second point is located seawards. There are 3 options to define the coordinates of the transects:
1. Interactively draw shore-normal transects along the mapped shorelines:
```
transects = SDS_transects.draw_transects(output, settings)
```
2. Load the transect coordinates from a .geojson file:
```
transects = SDS_tools.transects_from_geojson(path_to_geojson_file)
```
3. Create the transects by manually providing the coordinates of two points:
```
transects = dict([])
transects['Transect 1'] = np.array([[342836, ,6269215], [343315, 6269071]])
transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
```

**Note:** if you choose option 2 or 3, make sure that the points that you are providing are in the spatial reference system defined by `settings['output_epsg']`.

Once the shore-normal transects have been defined, the intersection between the 2D shorelines and the transects is computed with the following function:
```
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings)
```
The parameter `along_dist` defines the along-shore distance around the transect over which shoreline points are selected to compute the intersection. The default value is 25 m, which means that the intersection is computed as the median of the points located within 25 m of the transect (50 m alongshore-median). This helps to smooth out localised water levels in the swash zone.

An example is shown in the animation below:

![transects](https://user-images.githubusercontent.com/7217258/49990925-8b985a00-ffd3-11e8-8c54-57e4bf8082dd.gif)

### 2.4 Tidal correction <a class="anchor" id="section_2_4"></a>

Each satellite image is captured at a different stage of the tide, therefore a tidal correction is necessary to remove the apparent shoreline changes cause by tidal fluctuations.

`cross_distance = correct_tides.correct_tides(cross_distance,settings,output,reference_elevation,beach_slope)`

In order to tidally-correct the time-series of shoreline change you will need the following data:

* Time-series of water/tide level: this can be formatted as a .csv file, an example is provided [here](https://github.com/GIS-UNDP/coastline-change-prediction-tool/blob/main/data/GHANA/GHANA_tides.csv). Make sure that the dates are in UTC time as the shorelines are always in UTC time. Also the vertical datum needs to be approx. Mean Sea Level (MSL). If those tide values are in Mean Lower Low Water (MLLW), you will need to get the constant value of this datum at your station. 

`reference_elevation = 0` if tides are in MSL or `reference_elevation = MLLW value` otherwise.


* An estimate of the beach-face slope along each transect. If you don't have this data you can estimate it either using CoastSat.slope, see Vos et al. 2020 for more details (preprint available [here](https://www.essoar.org/doi/10.1002/essoar.10502903.1)) or using a global worldwide dataset of nearshore slopes estimates with a resolution of 1 km made by Athanasiou et al. 2019, see [their artice](https://essd.copernicus.org/articles/11/1515/2019/) for more details.

If you already have a beach slope estimate:

`beach_slope = 0.1
cross_distance = correct_tides.correct_tides(cross_distance,settings,output,reference_elevation,beach_slope)`

If you want to estimate the beach slope using  CoastSat.slope:

`cross_distance = correct_tides.correct_tides(cross_distance,settings,output,reference_elevation,estimate_slope=True)`

If you want to use the estimation made by Athanasiou et al. 2019:

`cross_distance = correct_tides.correct_tides(cross_distance,settings,output,reference_elevation)`

**Note**: if you don't have measured water levels, it is possible to obtain an estimate of the  time-series of modelled tide levels at the time of image acquisition from the [FES2014](https://www.aviso.altimetry.fr/es/data/products/auxiliary-products/global-tide-fes/description-fes2014.html) global tide model. Instructions on how to install the global tide model are available [here](https://github.com/kvos/CoastSat.slope/blob/master/doc/FES2014_installation.md).

The function `correct_tides` returns the tidally-corrected time-series of shoreline change and we call `reconstruct_shoreline` to recover the corrected shorelines and save them as shapefiles.

You will find several websites from where you can get tides prediction or covering the last months. In order to properly correct our shorelines, we need to have tide tables covering not only the last months but also the last years. There are several websites for this purpose but we only found one that can provide tides for long periods rather quickly. The steps to take are detailed below.  

**Extract the tides from the National Oceanic and Atmospheric Administration (NOAA)**

On the [NOAA website](https://tidesandcurrents.noaa.gov/tide_predictions.html), you will find tides tables for the USA, Carribean islands and some Pacific islands but not all of them.
You can get the tides annually from 2019 to 2021 or monthly for the years before 2019. If you have a lot of years to retrieve from the website, the fastest way to do it is to modify the downloading link below :
```
https://tidesandcurrents.noaa.gov/cgi-bin/predictiondownload.cgi?&stnid=STATION_ID&threshold=&thresholdDirection=&bdate=START_DATE&edate=END_DATE&units=metric&timezone=GMT&datum=MLLW&interval=hilo&clock=24hour&type=txt&annual=true
```
In this link, change the following :
* STATION_ID : You can find it in the name if the station you want to extract tides from (example : stnid=TEC4723 for Santo Domingo or stnid=TPT2707 for Taongi Atoll).
* START_DATE and END_DATE : You can only extract the years one by one, so you must change START_DATE and END_DATE by January 1st and December 31st with this format : yyyymmdd. For example, bdate=20010101 and edate=20011231 if you want to extract all the tides for the year 2001.  

Repeat this operation for every year you want to cover and paste the tides in a blank Excel file by using *Paste Special > Text*. Use the following formula in J1 and then [expand it](https://support.microsoft.com/en-us/office/copy-a-formula-by-dragging-the-fill-handle-in-excel-for-mac-dd928259-622b-473f-9a33-83aa1a63e218) to select only the valuable information from the tides tab :
```
=TEXT(A1;"yyyy-mm-dd") & " " & TEXT(C1;"hh:mm") & "," & SUBSTITUTE(F1/100;",";".")
```
Once you have expanded your formula, copy and paste your column in a new Excel file using the *Values* option when pasting. Then, save your file as .csv.

As the tide values retrieved on the NOAA website will be in Mean Lower Low Water (MLLW), you need to get the value of this datum at this station. For that, find your sation [here](https://tidesandcurrents.noaa.gov/stations.html?type=Datums) and read the value of the MLLW in the chart.

### 2.5 Computation of the prediction's generalization error <a class="anchor" id="section_2_5"></a>

You need to set up the `n_years_further` parameter to define the number of years for which the prediction will be performed.
The function `model_evaluation` computes the **generalization error** of the prediction by splitting the images into train and test samples and comparing the predicted shorelines with actual shorelines. It also computes the parameters that minimize the rmse and stores them in a `.pkl` so that the `predict` function reads the file and uses these parameters if it exists. This module delivers error metrics for each predicted year and best-estimated parameters for `Holt` and `SARIMAX` models. This gives an overvirew of model performances. It is still recommended to run the forecast for all three models and to consider both: error metrics and visual output. Model information can be found in section 2.6. Before starting the `model_evaluation` function, the `validate_year` function will check if the time series is long enough to provide the prediction for the inserted time period.

```
# define the number of years to be predicted
n_years_further = 2

message, validity = pt.validate_year(n_years_further, output)`

# model could be: Holt, ARIMA or SARIMAX
model = 'SARIMAX'
best_param, rmse, mae = pt.model_evaluation(cross_distance, n_years_further, metadata, output, settings, validity, model=model, smooth_data=False, smooth_coef=1, best_param=True, seasonality=1, MNDWI=False)
```

### 2.6 Time series forecasting <a class="anchor" id="section_2_6"></a>

Several models are implemented to predict the evolution of the time series for each transect. Beforehand, interpolation was carried out in order to make the intervals between the different steps regular. 

Time series data analysis means analysis of time series data to get the meaningful information from the data. Time series forecasting uses model to predict future values based on previous observed values at the present time. In other words, the physical parameters of the studied phenomenon, here the displacement of the coastline, are already present in the data and will therefore be taken into account in the prediction using a time series forecasting model.

The function `predict` allows to get the prediction and the `model` parameter allows to select the desired algorithm among:

* a **Holt's Linear Trend** model `'Holt'`

Exponential smoothing is one of the most preferred methods for a wide variety of time series data for its simplicity to understand, to implement with a simple numerical program, and for reliable forecast in a wide variety of applications.

Single exponential smoothing (SES) is a well-known method of forecasting for stationary time series data. However, it is not reliable when processing non-stationary time series data. Cross distances data have a general tendency of decreasing over time. So, they contain a trend and SES will not be useful in this case. Therefore, Holt gave a method to deal with data pertaining trend which is known as **Holt’s linear trend** method. Holt’s linear trend method comprises two smoothing constants, two smoothing equations and one forecast equation.

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\hat{y}_{t&plus;k}&space;&=&space;a_t&space;&plus;&space;kc_t&space;\\&space;a_t&space;&=&space;\gamma&space;y_t&space;&plus;&space;(1-\gamma)(a_{t-1}&plus;c_{t-1})&space;\\&space;c_t&space;&=&space;\delta&space;(a_t&space;-&space;a_{t-1})&space;&plus;&space;(1-\delta)c_{t-1}\\&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;\hat{y}_{t&plus;k}&space;&=&space;a_t&space;&plus;&space;kc_t&space;\\&space;a_t&space;&=&space;\gamma&space;y_t&space;&plus;&space;(1-\gamma)(a_{t-1}&plus;c_{t-1})&space;\\&space;c_t&space;&=&space;\delta&space;(a_t&space;-&space;a_{t-1})&space;&plus;&space;(1-\delta)c_{t-1}\\&space;\end{align*}" title="\begin{align*} \hat{y}_{t+k} &= a_t + kc_t \\ a_t &= \gamma y_t + (1-\gamma)(a_{t-1}+c_{t-1}) \\ c_t &= \delta (a_t - a_{t-1}) + (1-\delta)c_{t-1}\\ \end{align*}" /></a>

where (1) represents the forecast equation, (2) denotes the level equation and (3) represents the trend equation. 

γ and δ are smoothing constants for level and trend respectively whose values lie on the interval from 0 to 1.

a and c are estimates of the level and the trend of the time series respectively.

y denotes the observation.


The characteristics of Holt's linear trend method are mentioned below:
1. Holt's exponential smoothing method doesn't work with data that show cyclical or seasonal patterns. In our case, it should not be a problem as coastal shrinkage is not a cyclical physical phenomenon.
2. It is based to use for short term forecasting as it is based on the assumption that future trend will follow the current trend. This means that we will have to predict much less steps than we use to make the prediction. The longer you want to make a prediction, the older data you will need.
3. It does not provide good result in case of small data. In our case, if the first images are only three-four years old, the prediction will be rather bad. A range of images over at least seven years is more reasonable.

* an **AutoRegressive Integrated Moving Average model** `'ARIMA'`

ARIMA is a class of models that explains a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values. 

An ARIMA model is characterized by 3 terms: p, d, q

where,

p is the order of the AR term

q is the order of the MA term

d is the number of differencing required to make the time series stationary.

Potential pros of using ARIMA models:
- Only requires the prior data of a time series to generalize the forecast.
- Performs well on short term forecasts.
- Models non-stationary time series.


Potential cons of using ARIMA models:
- Difficult to predict turning points.
- There is quite a bit of subjectivity involved in determining (p,d,q) order of the model.
- Computationally expensive.
- Poorer performance for long term forecasts.
- Cannot be used for seasonal time series.

Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.

* an **Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors** `'SARIMAX'`

If your time series has defined seasonality, SARIMAX is a good model choice since it uses seasonal differencing.

Seasonal differencing is similar to regular differencing, but, instead of subtracting consecutive terms, you subtract the value from previous season.

So, the model will be represented as SARIMA(p,d,q)x(P,D,Q), where, P, D and Q are SAR, order of seasonal differencing and SMA terms respectively and 'x' is the frequency of the time series.

The SARIMAX model sometimes delivers LinAlgError. This is avoided using the exception, however, it is recommended to investigate this issue for the next version of the tool.

The function `predict` returns the time-series along each transect with the prediction for the `n_years_further` years added and a dates vector with the predicted dates added.

Some other parameters of the predict function can be specified:

* `smooth_data` (default to *True*): if set to True the training data is smoothed. This allows to erase recent outliers and at the same time to move back the period on which the learning is focused.
* `smooth_coef` (default to *1*): smoothing strength. The higher the value, the smoother the data.
* `seasonality` (default to *1*): number of periods in season.
* `plot`: if set to True, the time series predictions will be plotted.

```
time_series_pred, dates_pred_m = pt.predict(cross_distance,output,inputs,settings,n_years_further,validity,model=model,param=None,smooth_data=True,smooth_coef=1,seasonality=1,plot=True)
```
### 2.7 New shoreline reconstruction <a class="anchor" id="section_2_7"></a>

The function `reconstruct_shoreline` reconstructs the predicted shorelines. It keeps one shoreline per year predicted. These new shorelines will be plotted and saved as geojson and shapefiles.

```
predicted_sl = rs.reconstruct_shoreline(time_series_pred,transects,dates_pred_m,output,inputs,settings,n_years_further)
```

## Issues

Having a problem? Post an issue in the [Issues page](https://github.com/Space4Dev/coastline-predictor/issues).

## References

Kilian Vos, Kristen D. Splinter, Mitchell D. Harley, Joshua A. Simmons, Ian L. Turner,
CoastSat: A Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery,
Environmental Modelling & Software,
Volume 122,
2019,
104528,
ISSN 1364-8152,
https://doi.org/10.1016/j.envsoft.2019.104528.
(https://www.sciencedirect.com/science/article/pii/S1364815219300490)

Saha, Amit & Sinha, Kanchan. (2020). Usage of Holt's Linear Trend Exponential Smoothing for Time Series Forecasting in Agricultural Research. 
(https://www.researchgate.net/publication/345413376_Usage_of_Holt's_Linear_Trend_Exponential_Smoothing_for_Time_Series_Forecasting_in_Agricultural_Research)

Athanasiou, P., van Dongeren, A., Giardino, A., Vousdoukas, M., Gaytan-Aguilar, S., and Ranasinghe, R.: Global distribution of nearshore slopes with implications for coastal retreat, Earth Syst. Sci. Data, 11, 1515–1529, https://doi.org/10.5194/essd-11-1515-2019, 2019



## Issues
Having a problem? Post an issue in the [Issues page](https://github.com/kvos/coastsat/issues) (please do not email).

## Contributing
If you are willing to contribute, check out our todo list in the [Projects page](https://github.com/kvos/CoastSat/projects/1).
1. Fork the repository (https://github.com/kvos/coastsat/fork).
A fork is a copy on which you can make your changes.
2. Create a new branch on your fork
3. Commit your changes and push them to your branch
4. When the branch is ready to be merged, create a Pull Request (how to make a clean pull request explained [here](https://gist.github.com/MarcDiethelm/7303312))

## References

1. Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. *Environmental Modelling and Software*. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)

2. Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (2019). Sub-annual to multi-decadal shoreline variability from publicly available satellite imagery. *Coastal Engineering*. 150, 160–174. https://doi.org/10.1016/j.coastaleng.2019.04.004

3. Vos K., Harley M.D., Splinter K.D., Walker A., Turner I.L. (2020). Beach slopes from satellite-derived shorelines. *Geophysical Research Letters*. 47(14). https://doi.org/10.1029/2020GL088365 (Open Access preprint [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2))

4. Castelle B., Masselink G., Scott T., Stokes C., Konstantinou A., Marieu V., Bujan S. (2021). Satellite-derived shoreline detection at a high-energy meso-macrotidal beach. *Geomorphology*. volume 383, 107707. https://doi.org/10.1016/j.geomorph.2021.107707

5. Vos, K. and Deng, W. and Harley, M. D. and Turner, I. L. and Splinter, K. D. M. (2022). Beach-face slope dataset for Australia. *Earth System Science Data*. volume 14, 3, p. 1345--1357. https://doi.org/10.5194/essd-14-1345-2022

6. Training dataset used for pixel-wise classification in CoastSat: https://doi.org/10.5281/zenodo.3334147
