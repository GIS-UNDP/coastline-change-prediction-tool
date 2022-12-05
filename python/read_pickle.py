import pickle
import os

filepath = 'E:\\UNDP\\Coastline prediction tool\\new_data\\SDG_AI_Lab\\coastsat_dev\\data'
sitename = 'TARAWA'

with open(os.path.join(filepath,sitename,'mae_ARIMA.pkl'), 'rb') as f:
    data1 = pickle.load(f)

with open(os.path.join(filepath,sitename,'rmse_ARIMA.pkl'), 'rb') as k:
    data2 = pickle.load(k)

with open(os.path.join(filepath,sitename,'best_parameters_Holt.pkl'), 'rb') as k:
    data3 = pickle.load(k)

print('stop')


# filepath = 'C:\\Users\\Ivana\\Downloads'

# with open(os.path.join(filepath,'mae_Holt.pkl'), 'rb') as f:
#     data1 = pickle.load(f)

# with open(os.path.join(filepath,'rmse_Holt.pkl'), 'rb') as k:
#     data2 = pickle.load(k)

# with open(os.path.join(filepath,'best_parameters_Holt.pkl'), 'rb') as k:
#     data3 = pickle.load(k)

# print('stop')