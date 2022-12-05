import os
import numpy as np
import pandas as pd
import scipy.spatial
from coastsat import SDS_tools, SDS_slope
import matplotlib
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

def correct_tides(cross_distance,settings,output,reference_elevation=0,beach_slope=None,plot=True,estimate_slope=False):
    # load the measured tide data
    filepath = os.path.join(os.getcwd(),'data',settings['inputs']['sitename'],settings['inputs']['sitename']+'_tides.csv')
    tide_data = pd.read_csv(filepath, parse_dates=['dates'])
    dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
    tides_ts = np.array(tide_data['tide'])
    
    # get tide levels corresponding to the time of image acquisition
    dates_sat = output['dates']
    tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)
    
    if plot:
        # plot the subsampled tide data
        fig, ax = plt.subplots(1,1,figsize=(15,4), tight_layout=True)
        ax.grid(which='major', linestyle=':', color='0.5')
        ax.plot(tide_data['dates'], tide_data['tide'], '-', color='0.6', label='all time-series')
        ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w',lw=1, label='image acquisition')
        ax.set(ylabel='tide level [m]',xlim=[dates_sat[0],dates_sat[-1]], title='Water levels at the time of image acquisition');
        ax.legend()
    
    if estimate_slope:
        # settings for beach-face slopes estimation
        seconds_in_day = 24*3600
        settings_slope = {'slope_min':        0.000,                  # minimum slope to trial
                          'slope_max':        0.6,                    # maximum slope to trial
                          'delta_slope':      0.005,                  # slope increment
                          'date_range':       [2010,2021],            # range of dates over which to perform the analysis
                          'n_days':           8,                      # sampling period [days]
                          'n0':               50,                     # parameter for Nyquist criterium in Lomb-Scargle transforms
                          'freqs_cutoff':     1./(seconds_in_day*30), # 1 month frequency
                          'delta_f':          100*1e-10,              # deltaf for identifying peak tidal frequency band
                          }
        settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0],5,1)),
                                        pytz.utc.localize(datetime(settings_slope['date_range'][1],1,1))]
        beach_slopes = SDS_slope.range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])
        settings_slope['freqs_max'] = SDS_slope.find_tide_peak(dates_sat,tides_sat,settings_slope)
        
        # estimate beach-face slopes along the transects
        slope_est = dict([])
        for key in cross_distance.keys():
            # remove NaNs
            idx_nan = np.isnan(cross_distance[key])
            dates = [dates_sat[_] for _ in np.where(~idx_nan)[0]]
            tide = tides_sat[~idx_nan]
            composite = cross_distance[key][~idx_nan]
            # apply tidal correction
            tsall = SDS_slope.tide_correct(composite,tide,beach_slopes)
            title = 'Transect %s'%key
            SDS_slope.plot_spectrum_all(dates,composite,tsall,settings_slope, title)
            slope_est[key] = SDS_slope.integrate_power_spectrum(dates,tsall,settings_slope)
            print('Beach slope at transect %s: %.3f'%(key, slope_est[key]))
    elif beach_slope != None:
        slope_est = dict([])
        for key in cross_distance.keys():
            slope_est[key] = beach_slope
    else:
        filepath_slopes = os.path.join(os.getcwd(), 'nearshore_slopes.csv')
        slopes_data = pd.read_csv(filepath_slopes)
        
        x_array = np.array(slopes_data['X'])
        y_array = np.array(slopes_data['Y'])
        # Shoe-horn existing data for entry into KDTree routines
        points=np.array(settings['inputs']['polygon'][0][0])
        combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
        points_list = list(points[::-1])

        def do_kdtree(combined_x_y_arrays,points):
            mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
            dist, indexes = mytree.query(points)
            return indexes
        
        index = do_kdtree(combined_x_y_arrays,points_list)
        beach_slope = slopes_data.iloc[index]['slope']
        print(beach_slope)
        slope_est = dict.fromkeys(cross_distance.keys(),beach_slope)
        
    # tidal correction along each transect
    cross_distance_tidally_corrected = {}
    for key in cross_distance.keys():
        correction = (tides_sat-reference_elevation)/slope_est[key]
        cross_distance_tidally_corrected[key] = cross_distance[key] + correction
        
    # store the tidally-corrected time-series in a .csv file
    out_dict = dict([])
    out_dict['dates'] = dates_sat
    for key in cross_distance_tidally_corrected.keys():
        out_dict['Transect '+ key] = cross_distance_tidally_corrected[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                      'transect_time_series_tidally_corrected.csv')
    df.to_csv(fn, sep=',')
    print('Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s'%fn)
    
    if plot:
        # plot the time-series of shoreline change (both raw and tidally-corrected)
        fig = plt.figure(figsize=[15,8], tight_layout=True)
        gs = matplotlib.gridspec.GridSpec(len(cross_distance),1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        for i,key in enumerate(cross_distance.keys()):
            if np.all(np.isnan(cross_distance[key])):
                continue
            ax = fig.add_subplot(gs[i,0])
            ax.grid(linestyle=':', color='0.5')
            ax.set_ylim([-50,50])
            ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='raw')
            ax.plot(output['dates'], cross_distance_tidally_corrected[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='tidally-corrected')
            ax.set_ylabel('distance [m]', fontsize=12)
            ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
                    va='top', transform=ax.transAxes, fontsize=14)
        ax.legend()
    
    return cross_distance_tidally_corrected