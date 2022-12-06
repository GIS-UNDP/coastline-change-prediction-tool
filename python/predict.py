# -*- coding: utf-8 -*-
# %% load modules
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.interpolate import interp1d
from matplotlib import gridspec
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import ceil

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pmd
from pmdarima.arima import auto_arima

from pylab import ginput
from shapely import geometry
import geopandas as gpd
import skimage.transform as transform

from python import reconstruct_shoreline as rs
from python import analyze_shoreline as asl
from coastsat import SDS_tools, SDS_preprocess

from sklearn.model_selection import GridSearchCV
from pandas.plotting import autocorrelation_plot
from scipy import linalg

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print(linalg.lapack.dgetrf([np.nan]))
print(linalg.lapack.dgetrf([np.inf]))

#%% SUB-FUNCTIONS
# Replace nans and duplicates in cross_distance
def format_cd(cd,output):
    d=[]
    cross=[]
    for i in range(len(cd)):
        cond = np.isfinite(cd[i])
        if i<(len(cd)-1):
             cond = cond and output['dates'][i].date()!=output['dates'][i+1].date()
        if cond:
            d.append(output['dates'][i])
            cross.append(cd[i])
        elif i==0:
            d.append(output['dates'][i])
            cross.append(cd[np.where(np.isnan(cd)==False)[0][0]])
    return np.array(cross),np.array(d)

# Cross-distance with zeros where none
def fill_with_zeros(cd,dates_days,d):
    cdp=np.zeros(len(dates_days))
    idx=0
    for k in range(len(cdp)):     
        if idx<len(d) and dates_days[k].date()==d[idx].date():
            cdp[k]=cd[idx]
            idx+=1
    return cdp 

# Making the interpolation where zeros
def interpolate(cdp,cd):
    cdp2=np.copy(cdp)
    ind=np.argwhere(cdp2).flatten()
    f=interp1d(ind, cd)
    last=0
    for i in range(len(cdp)):
        if cdp2[i]==0:
            if i<=ind[-1]:
                cdp2[i]=f(i)
                last=i
            else:
                cdp2[i]=f(last)
    return cdp2

# Use only one image per month after interpolation
def filter_months(var,dates,cross_distance_pred=None):
    if var == 'distance':
        cdp=[cross_distance_pred[0]]
        for n in range(1,len(cross_distance_pred)):
            if dates[n].day==28:
                cdp.append(cross_distance_pred[n])
        return np.array(cdp)                
    elif var == 'dates':
        i = 0
        start=dates[i]
        while start.day > 28:
            i = i + 1
            start = dates[i] 
        
        end=dates[-1]+pd.DateOffset(months=1)
        d=pd.date_range(start, end,freq='m')      
        return d

# Smooth data in order to keep the decreasing trend in case there are outliers

def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        
# Add predicted dates in vector
def all_dates(dates,n_steps_further,cdp):
    start=dates[-1]+pd.DateOffset(months=1)
    end=start+pd.DateOffset(months=n_steps_further)
    future=pd.date_range(start, end, freq='m').tolist()
    dates_pred_months=dates.tolist()+future
    while len(dates_pred_months)!=len(cdp):
        dates_pred_months = dates_pred_months[:-1]
                    
    return dates_pred_months

# Split a given univariate sequence into multiple samples where each sample 
# has a specified number of time steps (`n_in, by default 3`) and the output 
# has also a specified number of time steps (`n_out, by default 1`).
def convert_data_univariate(data, n_in=3, n_out=1):
    X, y = [], []
    # input sequence (t-n, ... t-1)
    for i in range(len(data)):
        # find the end of this pattern
        end_x = i + n_in
        end_y = end_x + n_out
        # check if we are beyond the sequence
        if end_y > len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_x], data[end_x:end_y]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# %% MODELS ALGORITHMS

# Holt’s Linear Trend algorithm
def predict_Holt_Trend(n_steps_further,cross_distance_pred,dates,output):
    df_ts = pd.DataFrame(cross_distance_pred)
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
    smooth_params = {'smoothing_level':0.2,'smoothing_slope':0.1}
    # fit model        
    fit = Holt(np.asarray(df_ts),exponential=True).fit(smoothing_level = smooth_params['smoothing_level'],smoothing_slope = smooth_params['smoothing_slope'])
    # make prediction
    y_hat = fit.forecast(n_steps_further)
    return y_hat

def predict_ARIMA(n_steps_further,cross_distance_pred,dates):
    def arimamodel(timeseriesarray):
        autoarima_model = pmd.auto_arima(timeseriesarray, 
                              start_p=1, 
                              start_q=2,
                              d=1, 
                              seasonal=True,
                              trend='t',
                              method='nm',
                              test="adf",
                              trace=True)
        return autoarima_model

    df_ts = pd.DataFrame(cross_distance_pred)
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
  
    years = int(n_steps_further / 12)

    start_date = df_ts.index[-1]+pd.DateOffset(months=1)
    end_date = df_ts.index[-1]+pd.DateOffset(years=years)
    start_date = start_date.date()
    end_date = end_date.date()
    
    # make prediction
    automodel = arimamodel(df_ts)
    ARIMA_predict_diff = automodel.predict(n_steps_further)
    ARIMA_predict_numpy = ARIMA_predict_diff.to_numpy()

    return ARIMA_predict_numpy

def predict_SARIMAX(n_steps_further,cross_distance_pred,dates,seasonality):    
    df_ts = pd.DataFrame(cross_distance_pred)
    auto_arima_model = auto_arima(y=df_ts,
                              seasonal=True,
                              m=seasonality,
                              information_criterion="aic",
                              trace=True)

    try:
        SARIMAXmodel = SARIMAX(endog=df_ts,order=auto_arima_model.order,seasonal_order=auto_arima_model.seasonal_order, trend='t').fit()
    except np.linalg.LinAlgError as err:
        # if type(err) == <class 'numpy.linalg.LinAlgError'>:
            return None
            
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
    y_pred = SARIMAXmodel.get_forecast(n_steps_further)
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])

    return y_pred_df['Predictions'].values

def validate_year(n_years_further, output):
    if type(n_years_further) is int:
        pass
    else:
        return print('Please insert a whole number.'), False

    start = output['dates'][0].strftime("%m/%d/%Y")
    end = output['dates'][-1].strftime("%m/%d/%Y")
    dates=pd.date_range(start, end).tolist()

    if n_years_further/(len(dates)/30/12) < 0.5:
        return print('Valid number of years.'), True
    else:
        return print('Timeseries is not long enough to predict coast position {} years in the future. Please choose a lower value or recalcuate the timeseris for a longer time period.'.format(n_years_further)), False

# get the results of the prediction using various models
def predict(cross_distance,output,inputs,settings,n_steps_further,validity, model='ARIMA',param=None,smooth_data=False,smooth_coef=5,tuning=True,seasonality=2,plot=True):
    if validity is False:
        raise Exception("Invalid number of years.")
    
    cd = cross_distance.copy()

    n_steps_further = n_steps_further * 12
    
    parameters = dict([])
    # Format dates
    start=output['dates'][0].strftime("%m/%d/%Y")
    end=output['dates'][-1].strftime("%m/%d/%Y")
    dates=pd.date_range(start, end).tolist()
    dates_days=np.copy(dates)
    
    dates=filter_months(dates=dates_days,var='dates')
    
    time_series_pred=dict([])

    if model == 'Holt':
        if os.path.exists(os.path.join(inputs['filepath'],inputs['sitename'],'best_parameters_%s.pkl'%model)):
            f= open(os.path.join(inputs['filepath'],inputs['sitename'],'best_parameters_%s.pkl'%model), 'rb') 
            params = pickle.load(f)
            f.close()

    elif model == 'SARIMAX':
        if os.path.exists(os.path.join(inputs['filepath'],inputs['sitename'],'best_parameters_%s.pkl'%model)):
            f= open(os.path.join(inputs['filepath'],inputs['sitename'],'best_parameters_%s.pkl'%model), 'rb') 
            params = pickle.load(f)
            f.close()
            seasonality = params

    for i in cd.keys():
        parameters[i] = {'order':None}
        cd[i],d=format_cd(cd[i],output)
        cross_distance_pred=fill_with_zeros(cd[i],dates_days,d)
        cross_distance_pred=interpolate(cross_distance_pred,cd[i])
        cross_distance_pred=filter_months(var='distance',dates=dates_days,cross_distance_pred=cross_distance_pred)

        if smooth_data and model == 'Holt':
            sc =int(smooth_coef) 
            if os.path.exists(os.path.join(inputs['filepath'],inputs['sitename'],'best_parameters_%s.pkl'%model)):
                # sc = int(params[i]['smooth_coef'])
                sc = int(params) 
                cdp_smoothed = smooth(np.copy(cross_distance_pred),sc)
            else:
                cdp_smoothed = cross_distance_pred
            
        ################# HOLT'S LINEAR TREND MODEL ########################
        if model == 'Holt':
            cross_distance_pred=np.concatenate((cross_distance_pred,predict_Holt_Trend(n_steps_further,cdp_smoothed,dates,output)))
        ######################## ARIMA MODEL ##############################
        if model == 'ARIMA':
            cross_distance_pred=np.concatenate((cross_distance_pred,predict_ARIMA(n_steps_further,cross_distance_pred,dates)))
        ################# SARIMAX LINEAR TREND MODEL ########################
        if model == 'SARIMAX':
            error = predict_SARIMAX(n_steps_further,cross_distance_pred,dates,seasonality)
            if type(error) == type(None):
                return None, None
            else:
                cross_distance_pred=np.concatenate((cross_distance_pred,predict_SARIMAX(n_steps_further,cross_distance_pred,dates,seasonality))) 
          
        time_series_pred[i]=np.array(cross_distance_pred)
        
    # Add predicted dates in vector
    dates_pred_months = all_dates(dates,n_steps_further,cross_distance_pred)

    if plot:
        maxi=[]
        for k in time_series_pred.keys():
           maxi.append(np.max(time_series_pred[k]))
        maxi=np.max(maxi)
        
        # Plot the new time series containing the prediction
        fig, ax = plt.subplots(figsize=[15,8])
        gs = gridspec.GridSpec(len(time_series_pred),1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        ax.grid(linestyle=':', color='0.5')
        ax.vlines(dates[-1],0,maxi,linestyle='dashed')
        l, = plt.plot(dates_pred_months, time_series_pred['1'], '-o', ms=6, mfc='w')
        ax.set_ylabel('Cross-distance (in metres)')
        ax.margins(x=0)   
        
        axcolor = 'lightgray'
        axtrans = plt.axes([0.18, 0.025, 0.65, 0.04], facecolor=axcolor)      
        strans = Slider(axtrans, 'Transect', 1, len(list(time_series_pred.keys())), valinit=1, valstep=1)
        
        def update(val):
            trans = strans.val
            l.set_ydata(time_series_pred[str(trans)])
            fig.canvas.draw_idle()
        
        strans.on_changed(update)
        plt.title('Time series with predictions for each transect',pad=400)
        plt.show() 
    
    return time_series_pred,dates_pred_months

# %% MODEL VALIDATION

def model_evaluation(cross_distance,  n_steps_further, metadata, output, settings, validity, model='Holt', smooth_data=True, smooth_coef=0, best_param=False, seasonality= 2,  MNDWI=False):
    if validity is False:
        raise Exception("Invalid number of years.")

    n_months_further = n_steps_further * 12
    param_sc = np.linspace(0,40,21)
    season_sc = np.arange(2,13)
    params = dict([])
    params_temp = dict([])
    rmse_all = dict([])
    mae_all = dict([])
    rmse_all_temp = dict([])
    mae_all_temp = dict([])
    yearly_error = dict([])
    years_list = []
    rmse_list = []
    
    if model=='Holt':
        min_error=100**10
        for i in range(len(param_sc)):
            
            for transect in cross_distance.keys():
                params_temp[transect] = {'smooth_coef':None}
                           
                error = predict_estimate_validate(cross_distance, n_months_further,transect,output,smooth_data=True,smooth_coef=int(param_sc[i])+1,model=model,params=None,tuned_parameters=None, lags=None, seasonality= 1, transformation=None)
                
                rmse_all_temp[transect + ' average per year'] = error[0]
                rmse_all_temp[transect] = error[2]
                mae_all_temp[transect + ' average per year'] = error[1]
                mae_all_temp[transect] = error[3]

            rmse_list = list(rmse_all_temp.values())[1::2] # all RMSE values
            if min_error > np.mean(rmse_list):
                min_error = np.mean(rmse_list)

                params = param_sc[i]+1
                rmse_all = rmse_all_temp
                mae_all = mae_all_temp

        for n_transects in range(len(cross_distance.keys())):
            years_list.extend(list(mae_all[list(mae_all.keys())[0::2][n_transects]].values()))
        
        n_years = len(mae_all[list(mae_all.keys())[0::2][0]].keys())

        count = 0
        for x in range(n_years):
            yearly_error[x] = np.mean(years_list[count::n_years])
            count = count + 1
            print('Generalization error (mean mae) for the year ',  str(x+1) , ' in the future: ' , yearly_error[x],'\n')

    elif model=='ARIMA':
        for transect in cross_distance.keys():
                      
            error = predict_estimate_validate(cross_distance, n_months_further,transect,output,smooth_data=False,smooth_coef=1,model=model,params=None,tuned_parameters=None, lags=None, seasonality= 1, transformation=None)
                           
            rmse_all[transect + ' average per year'] = error[0]
            rmse_all[transect] = error[2]
            mae_all[transect + ' average per year'] = error[1]
            mae_all[transect] = error[3]

        for n_transects in range(len(cross_distance.keys())):
            years_list.extend(list(mae_all[list(mae_all.keys())[0::2][n_transects]].values()))
        
        n_years = len(mae_all[list(mae_all.keys())[0::2][0]].keys())

        count = 0
        for x in range(n_years):
            yearly_error[x] = np.mean(years_list[count::n_years])
            count = count + 1
            print('Generalization error (mean mae) for the year ',  str(x+1) , ' in the future: ' , yearly_error[x],'\n')
       
    elif model=='SARIMAX':
        min_error=100**10
        for i in range(len(season_sc)):
            
            for transect in cross_distance.keys():
                params_temp[transect] = {'seasonality':None}
                           
                error = predict_estimate_validate(cross_distance, n_months_further,transect,output,smooth_data=False,smooth_coef=1,model=model,params=None,tuned_parameters=None, lags=None, seasonality= season_sc[i],transformation=None)
                
                if error == None:
                    break
                else:
                    rmse_all_temp[transect + ' average per year'] = error[0]
                    rmse_all_temp[transect] = error[2]
                    mae_all_temp[transect + ' average per year'] = error[1]
                    mae_all_temp[transect] = error[3]

            rmse_list = list(rmse_all_temp.values())[1::2] # all RMSE values
            if error == None:
                continue
            elif min_error > np.mean(rmse_list):
                min_error = np.mean(rmse_list)

                params = season_sc[i]
                rmse_all = rmse_all_temp
                mae_all = mae_all_temp

        for n_transects in range(len(cross_distance.keys())):
            years_list.extend(list(mae_all[list(mae_all.keys())[0::2][n_transects]].values()))
        
        n_years = len(mae_all[list(mae_all.keys())[0::2][0]].keys())

        count = 0
        for x in range(n_years):
            yearly_error[x] = np.mean(years_list[count::n_years])
            count = count + 1
            print('Generalization error (mean mae) for the year ',  str(x+1) , ' in the future: ' , yearly_error[x],'\n')
    
    f = open(os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'rmse_%s.pkl'%model), 'wb') 
    pickle.dump(rmse_all, f)
    f.close()
    
    # Save mae in .pkl to retrieve it in estimate_pop if exists
    f = open(os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'mae_%s.pkl'%model), 'wb') 
    pickle.dump(mae_all, f)
    f.close()

    # Save best parameters in .pkl to retrieve them in predict function if exist
    if best_param is True:
        f = open(os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'best_parameters_%s.pkl'%model), 'wb') 
        pickle.dump(params, f)
        f.close()
        return params, rmse_all, mae_all
    else:
        return None, rmse_all, mae_all

def predict_estimate_validate(cross_distance, n_months_further,transect,output,smooth_data=True,smooth_coef=30,model='Holt',params=None,tuned_parameters=None, lags=None, seasonality=2, transformation=None):
    cd = cross_distance.copy()
    
    # Format dates
    start=output['dates'][0].strftime("%m/%d/%Y")
    end=output['dates'][-1].strftime("%m/%d/%Y")
    dates=pd.date_range(start, end).tolist()
    dates_days=np.copy(dates)
    
    dates=filter_months(var='dates',dates=dates_days)
      
    # Make the prediction and store it in a similar dict named 'time_series_pred'
    rmse = dict({})
    mae = dict({})
    rmse['mean'] = dict({})
    mae['mean'] = dict({})
    
    # cross_dist_keys = list(cross_distance.keys())
      
    cd[transect],d=format_cd(cd[transect],output)
    cross_distance_pred=fill_with_zeros(cd[transect],dates_days,d)
    cross_distance_pred=interpolate(cross_distance_pred,cd[transect])
    cross_distance_pred=filter_months(var='distance',dates=dates_days,cross_distance_pred=cross_distance_pred)        

    # Smooth data
    if smooth_data:
        cdp_smoothed = smooth(cross_distance_pred,smooth_coef)
    else:
        cdp_smoothed = cross_distance_pred

    n_steps_further = n_months_further
    # split into train and test sets
    SPLIT = n_steps_further 
    X_train = cdp_smoothed[:-SPLIT]
    y_test = cdp_smoothed[-SPLIT:]


    ################# HOLT'S LINEAR TREND MODEL ########################
    if model == 'Holt':
        cross_distance_pred2 = predict_Holt_Trend(n_steps_further,X_train,dates[:-SPLIT],output)
    ################# SARIMAX LINEAR TREND MODEL ########################
    if model == 'SARIMAX':
        cross_distance_pred2 = predict_SARIMAX(n_steps_further,X_train,dates[:-SPLIT],seasonality)
        if type(cross_distance_pred2) == type(None):
            return None

    ######################## ARIMA MODEL ##############################
    if model == 'ARIMA':
        cross_distance_pred2 = predict_ARIMA(n_steps_further,X_train,dates[:-SPLIT])

    # Add predicted dates in vector
    dates_pred_m = all_dates(dates[-SPLIT:],n_steps_further, cross_distance_pred2)

    # Format dates
    start=dates[-SPLIT:][0].strftime("%m/%d/%Y")
    end=dates[-SPLIT:][-1].strftime("%m/%d/%Y")
    dates_all=pd.date_range(start, end).tolist()
    
    cross_distance_pred_int = fill_with_zeros(cross_distance_pred2, dates_all, dates_pred_m)
    cross_distance_pred = interpolate(cross_distance_pred_int,cross_distance_pred2)

    y_test_ref_int = fill_with_zeros(y_test, dates_all,  dates_pred_m)
    y_test_ref = interpolate(y_test_ref_int, y_test)

    cross_distance_pred_df = pd.DataFrame(cross_distance_pred,index=dates_all)
    y_test_ref_df = pd.DataFrame(y_test_ref,index=dates_all)

    # residuals = cross_distance_pred_df.subtract(y_test_ref_df)

    count = 0
    rmse = dict({})
    mae = dict({})

    mae_overall = mean_absolute_error(y_test_ref_df, cross_distance_pred_df)
    rmse_overall = np.sqrt(mean_squared_error(y_test_ref_df, cross_distance_pred_df))

    count = 0
    n_years = dates_all[-1].year - dates_all[0].year
    first_year = dates_all[0].year
    # n_dates = 0
    while count < (n_years + 1):
        year = first_year + count
        cross_distance_pred_residuals_yearly = cross_distance_pred_df.loc[cross_distance_pred_df.index.year == year]
        y_test_ref_df_yearly = y_test_ref_df.loc[y_test_ref_df.index.year == year]

        if len(cross_distance_pred_residuals_yearly) < 15:
            count = count + 1
            continue
        else:
            pass

        mae['year ' + str(year)] = mean_absolute_error(y_test_ref_df_yearly, cross_distance_pred_residuals_yearly) 
        rmse['year ' + str(year)] = np.sqrt(mean_squared_error(y_test_ref_df_yearly, cross_distance_pred_residuals_yearly)) 
        count = count + 1
                    
    return rmse, mae, rmse_overall, mae_overall