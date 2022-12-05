# -*- coding: utf-8 -*-
# %% load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import gridspec
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima_model import ARIMA
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import GridSearchCV
# %%

# %% SUB-FUNCTIONS
# Replace nans in cross_distance
def format_cd(cd,output):
    d=[]
    cross=[]
    for i in range(len(cd)):
        if np.isfinite(cd[i]):
            d.append(output['dates'][i])
            cross.append(cd[i])
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
            if dates[n].day==1:
                cdp.append(cross_distance_pred[n])
        return np.array(cdp)                
    elif var == 'dates':
        start=dates[0]
        end=dates[-1]+pd.DateOffset(months=1)
        d=pd.date_range(start, end,freq='m')      
        return d

# Add predicted dates in vector
def all_dates(dates,n_steps_further):
    start=dates[-1]+pd.DateOffset(months=1)
    end=start+pd.DateOffset(months=n_steps_further)
    future=pd.date_range(start, end, freq='m').tolist()
    dates_pred_months=dates.tolist()+future
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
# %%

# %% CROSS-VALIDATION
def global_weighted_error(err):
    mean = 0
    coeffs = 0
    for i in err.keys():
        if int(i) <= 3:
            mean += 2*err[i]
            coeffs += 2
        else:
            mean += err[i]
            coeffs += 1
    mean = mean/coeffs
    return mean

def cross_validation(cross_distance,output,model='Holt',tuned_parameters=None):
    ################### AUTOREGRESSIVE MODEL ###########################
    if model == 'AR':
        print('')
        #error = predict_test(cross_distance,output,model='AR',tuned_parameters=None)
    ################# HOLT'S LINEAR TREND MODEL ########################
    if model == 'Holt':
        param_sl = np.linspace(0,1,20)
        param_ss = np.linspace(0,1,20)
        min_error=10**10
        param = {'smoothing_level':None,'smoothing_slope':None}
        for i in range(len(param_sl)):
            for j in range(len(param_ss)):
                error = predict_test(cross_distance,output,model='Holt',tuned_parameters=None)
                mae = global_weighted_error(error[1]['mean'])
                if rmse < min_error:
                    min_error = mae
                    param['smoothing_level'] = param_sl[i]
                    param['smoothing_slope'] = param_ss[j]
        print('Generalization error (mean rmse) : ',min_error,'\n')
    ######################## ARIMA MODEL ##############################
    if model == 'ARIMA':
        print('')
        #error = predict_test(cross_distance,output,model='ARIMA',tuned_parameters=None)
    #################### STACKED BI-LSTM MODEL #########################
    if model == 'LSTM':
        print('')
        #error = predict_test(cross_distance,output,model='LSTM',tuned_parameters=None)
    return param 
# %%

# %% MODELS ALGORITHMS
# AutoRegressive model
def predict_AR(n_steps_further,cross_distance_pred,dates,tuned_parameters):
    transformation,p,d,_ = tuned_parameters
    # apply a transformation to remove the trend
    df_ts = pd.DataFrame(cross_distance_pred)
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
    df_trans = transformation(df_ts)
    
    # fit model
    model = ARIMA(df_trans.T.iloc[0], order=(p, d, 0))
    results_AR = model.fit(disp=-1)  
    start_date = df_ts.index[-1]+pd.DateOffset(months=1)
    end_date = df_ts.index[-1]+pd.DateOffset(years=1)
    start_date = start_date.date()
    end_date = end_date.date()
    
    # make prediction
    AR_predict_diff = results_AR.predict(start=start_date,end=end_date)
    
    # apply inverse transformation
    AR_predict_numpy = AR_predict_diff.to_numpy()
    x = np.linspace(np.min(AR_predict_numpy),np.max(AR_predict_numpy),len(AR_predict_numpy))
    y_hat = np.zeros(AR_predict_numpy.shape)
                     
    def diff(x,a):
        yt = transformation(x)
        return (yt - a )**2

    for idx,x_value in enumerate(x):
        res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
        y_hat[idx] = res.x[0]

    cross_distance_pred=np.concatenate((cross_distance_pred,y_hat))
    return cross_distance_pred

# Holt’s Linear Trend algorithm
def predict_Holt_Trend(n_steps_further,cross_distance_pred,dates,output):
    df_ts = pd.DataFrame(cross_distance_pred)
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
    # fit model
    #params = cross_validation(cross_distance,output,model='Holt')
    params = {'smoothing_level':0.2,'smoothing_slope':0.1}
    fit = Holt(np.asarray(df_ts),exponential=True).fit(smoothing_level=params['smoothing_level'],smoothing_slope=params['smoothing_slope'])
    # make prediction
    y_hat = fit.forecast(n_steps_further)
    cross_distance_pred = np.concatenate((cross_distance_pred,y_hat))
    return cross_distance_pred

def predict_ARIMA(n_steps_further,cross_distance_pred,dates,tuned_parameters):
    transformation,p,d,q = tuned_parameters
    # apply a transformation to remove the trend
    df_ts = pd.DataFrame(cross_distance_pred)
    df_ts.index = pd.to_datetime(dates, format='%d-%m-%Y %H:%M')
    df_trans = transformation(df_ts)
    
    # fit model
    model = ARIMA(df_trans.T.iloc[0], order=(p, d, q))
    results_ARIMA = model.fit(disp=-1)  
    start_date = df_ts.index[-1]+pd.DateOffset(months=1)
    end_date = df_ts.index[-1]+pd.DateOffset(years=1)
    start_date = start_date.date()
    end_date = end_date.date()
    
    # make prediction
    ARIMA_predict_diff = results_ARIMA.predict(start=start_date,end=end_date)
    
    # apply inverse transformation
    ARIMA_predict_numpy = ARIMA_predict_diff.to_numpy()
    x = np.linspace(np.min(ARIMA_predict_numpy),np.max(ARIMA_predict_numpy),len(ARIMA_predict_numpy))
    y_hat = np.zeros(ARIMA_predict_numpy.shape)
                     
    def diff(x,a):
        yt = transformation(x)
        return (yt - a )**2

    for idx,x_value in enumerate(x):
        res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
        y_hat[idx] = res.x[0]

    cross_distance_pred=np.concatenate((cross_distance_pred,y_hat))
    return cross_distance_pred

def predict_LSTM(n_steps_further,cross_distance_pred):
    n_features = 1 # for univariate time series
    
    LSTM_SIZE = 16
    
    # split into samples
    n_in = 12
    n_out = 1

    X, y = convert_data_univariate(cross_distance_pred, n_in, n_out)

    # Stacked Bi-LSTM Model
    inputs = Input(shape=(n_in, n_features))
    outputs = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, activation='relu'))(inputs)
    outputs = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, activation='relu'))(outputs)
    outputs = Bidirectional(LSTM(LSTM_SIZE, return_sequences=False, activation='relu'))(outputs)
    predictions = Dense(n_out, activation='linear')(outputs)
   
    y_pred_array = []


    # reshape from [samples, timesteps] into [samples, n_in, n_features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    X_pred = np.array([X[-1]])
    
    for n in range(n_steps_further):
        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=inputs, outputs=predictions)    
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        mc = keras.callbacks.ModelCheckpoint("model.h5", monitor="val_mae", mode="min", save_best_only=True)
        callbacks = [es, mc]
    
        testAndValid = 0.1
        SPLIT = int(testAndValid*len(X))
        
        # fit model
        model.fit(X[:-SPLIT], y[:-SPLIT], epochs=200,
                            validation_data=(X[-SPLIT:], y[-SPLIT:]),
                            callbacks=callbacks, verbose=0, shuffle=False)
        
        # model prediction
        model = keras.models.load_model("model.h5")
        y_pred = model.predict(X_pred)

        y_pred = y_pred.reshape(-1)

        y_pred_array.append(y_pred[0])
        
        X_pred[:,:-1,:] = X_pred[:,1:,:]
        X_pred[:,-1,:] = y_pred[0]
        
    return np.concatenate((cross_distance_pred,np.array(y_pred_array)))
# %%

# %% ERROR
# get the generalization error for our predictive models
def predict_test(cross_distance,cross_distance_ref,output,model='Holt',tuned_parameters=None):
    
    # Format dates
    start=output['dates'][0].strftime("%m/%d/%Y")
    end=output['dates'][-1].strftime("%m/%d/%Y")
    dates=pd.date_range(start, end).tolist()
    dates_days=np.copy(dates)
    
    dates=filter_months(var='dates',dates=dates_days)
    
    # get the indice to split in train/test samples
    start_date_ref = cross_distance_ref['dates'][0]
    count = 0
    found = False
    while not found and count<len(dates):
        if start_date_ref.month == dates[count].month:
            found = True
        else: 
            count += 1
    
    # Make the prediction and store it in a similar dict named 'time_series_pred'
    rmse = dict({})
    mae = dict({})
    rmse['mean'] = dict({})
    mae['mean'] = dict({})
    
    for i in cross_distance.keys():
        cross_distance_pred=fill_with_zeros(cross_distance[i],dates_days,output)
        cross_distance_pred=interpolate(cross_distance_pred,cross_distance[i])
        cross_distance_pred=filter_months(var='distance',dates=dates_days,cross_distance_pred=cross_distance_pred)        
        
        # split into train and test sets
        SPLIT = count
        X_train = cross_distance_pred[:SPLIT]
        y_test = cross_distance_pred[SPLIT:]
        
        n_steps_further = len(y_test)
        
        ################### AUTOREGRESSIVE MODEL ###########################
        if model == 'AR':
            cross_distance_pred=predict_AR(n_steps_further,X_train,dates[:SPLIT],tuned_parameters[i][:SPLIT])
        ################# HOLT'S LINEAR TREND MODEL ########################
        if model == 'Holt':
            cross_distance_pred=predict_Holt_Trend(n_steps_further,X_train,dates[:SPLIT])
        ######################## ARIMA MODEL ##############################
        if model == 'ARIMA':
            cross_distance_pred=predict_ARIMA(n_steps_further,X_train,dates[:SPLIT],tuned_parameters[i][:SPLIT])
        #################### STACKED BI-LSTM MODEL #########################
        if model == 'LSTM':
            cross_distance_pred=predict_LSTM(n_steps_further,X_train)
        
        cross_distance_pred = fill_with_zeros(cross_distance_pred,dates_days,output)
        cross_distance_pred = interpolate(cross_distance_pred,cross_distance[i])
        
        rmse_temp = 0
        mae_temp = 0
        count = 0
        rmse[i] = dict({})
        mae[i] = dict({})
        for j,day in enumerate(dates_days):
            if day in cross_distance_ref['dates']:
                rmse_temp += (cross_distance_pred[j]-cross_distance_ref[i][count])**2
                mae_temp += abs(cross_distance_pred[j]-cross_distance_ref[i][count])
                count += 1 
            
            if j%365 == 0:
                # compute the prediction Root Mean Square Error and the Mean Absolute Error for each predicted year for the transect i
                rmse[i]['%i' %(j//365)] = sqrt(rmse_temp)
                mae[i]['%i' %(j//365)] = mae_temp/count
                
                rmse['mean']['%i' %(j//365)] += rmse_temp
                mae['mean']['%i' %(j//365)] += mae_temp/count
         
    for year in rmse['mean'].keys():
        rmse['mean'][year] = sqrt(rmse['mean'][year])
        mae['mean'][year] =  mae['mean'][year]/len(cross_distance.keys())
                
    return rmse, mae
# %%
        
# get the results of the prediction using various models
def predict(cross_distance,output,n_steps_further,model='Holt',tuned_parameters=None,plot=True):
    
    # Format dates
    start=output['dates'][0].strftime("%m/%d/%Y")
    end=output['dates'][-1].strftime("%m/%d/%Y")
    dates=pd.date_range(start, end).tolist()
    dates_days=np.copy(dates)
    
    dates=filter_months(dates=dates_days,var='dates')
    
    time_series_pred=dict([])
    # Make the prediction and store the RMSE in a list named 'rmse'
    for i in cross_distance.keys():
        cross_distance[i],d=format_cd(cross_distance[i],output)
        cross_distance_pred=fill_with_zeros(cross_distance[i],dates_days,d)
        cross_distance_pred=interpolate(cross_distance_pred,cross_distance[i])
        cross_distance_pred=filter_months(var='distance',dates=dates_days,cross_distance_pred=cross_distance_pred)
        ################### AUTOREGRESSIVE MODEL ###########################
        if model == 'AR':
            cross_distance_pred=predict_AR(n_steps_further,cross_distance_pred,dates,tuned_parameters[i])
        ################# HOLT'S LINEAR TREND MODEL ########################
        if model == 'Holt':
            cross_distance_pred=predict_Holt_Trend(n_steps_further,cross_distance_pred,dates,output)
        ######################## ARIMA MODEL ##############################
        if model == 'ARIMA':
            cross_distance_pred=predict_ARIMA(n_steps_further,cross_distance_pred,dates,tuned_parameters[i])
        #################### STACKED BI-LSTM MODEL #########################
        if model == 'LSTM':
            cross_distance_pred=predict_LSTM(n_steps_further,cross_distance_pred)
    
        time_series_pred[i]=np.array(cross_distance_pred)
    
    # Add predicted dates in vector
    dates_pred_months = all_dates(dates,n_steps_further)
    
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
        ax.vlines(output['dates'][-1],0,maxi,linestyle='dashed')
        l, = plt.plot(dates_pred_months[:len(time_series_pred['1'])], time_series_pred['1'], '-o', ms=6, mfc='w')
        ax.margins(x=0)   
        
        axcolor = 'lightgray'
        axtrans = plt.axes([0.18, 0.025, 0.65, 0.04], facecolor=axcolor)      
        strans = Slider(axtrans, 'Transect', 1, len(list(time_series_pred.keys())), valinit=1, valstep=1)
        
        def update(val):
            trans = strans.val
            l.set_ydata(time_series_pred[str(trans)])
            fig.canvas.draw_idle()
        
        strans.on_changed(update)
        plt.show() 
    
    return time_series_pred,dates_pred_months
