# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:07:05 2022

@author: Ra, S.J., Cho, S.
"""

import pandas as pd
import numpy as np
import sklearn
import os

import matplotlib.pyplot as plt
import joblib
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

os.chdir(r'C:\Users\YourPCPC\Desktop\DS_MetaLr')
#%% ANN function
def build_ANN(output_size, input_size, hidden_size):
    ann_ = Sequential()
    ann_.add(Input(input_size))
    for nodes in hidden_size:
        ann_.add(Dense(nodes, activation = 'relu',
                       kernel_initializer = RandomNormal(stddev=0.005),
                       bias_initializer = RandomNormal(stddev=0.005)))

    ann_.add(Dense(output_size))
    
    ann_.compile(
                 loss='mean_squared_error',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=['RootMeanSquaredError', 'MeanAbsoluteError'])
    return ann_

def cv_rmse(mea_y, pred_y):
    rmse_ = np.sqrt(np.mean(np.power((mea_y - pred_y), 2), axis=0))
    cvrmse_ = rmse_/np.mean(mea_y, axis=0)*100
    return cvrmse_

def mbe(mea_y, pred_y):
    mbe_ = np.sum((mea_y - pred_y), axis=0)/np.mean(mea_y, axis=0)/(len(mea_y)-1)*100
    return mbe_

#%% make dir function
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder('./Model/transferredModel')

#%%
os.chdir('./ANN_model')

"1. data load"
# data pre-process
# 실내/외기온도
z2temp = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_meta.xlsx",'Doosan_Testo_2105-06', engine='openpyxl', index_col='Time')
z2temp_sa = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_valid_meta.xlsx",'Sheet1', engine='openpyxl', index_col='Time')

# AHU 가동상태
z2ctrl = pd.read_excel(r"Doosan_AHU_2105-06_meta_vs2.xlsx",'on_off', engine='openpyxl', index_col='Time')
z2temp.index = z2temp.index.round('min')
z2temp_sa.index = z2temp_sa.index.round('min')

# retime
z2temp=z2temp.loc["2021-07-30":"2021-09-17",:]
z2ctrl=z2ctrl.loc["2021-07-30":"2021-09-17",:]
z2temp_sa=z2temp_sa.loc["2021-07-30":"2021-09-17",:]
# %%
## make input format per model
dat01_sa = pd.concat([z2ctrl[['AHU_CDU_SUM','AHU_FAN_SUM']],z2temp[['1-122[℃]','1-122[%]','1-142[℃]_급기온도']]], axis=1)  # z3: 1차 실증 구역 #1
dat01_sa = dat01_sa.dropna(axis=0)
## accumulate over time: (t) -> (t, t-10min)
tmp_1_sa = dat01_sa.shift(-10)
tmp_1_sa.columns = tmp_1_sa.columns + '_10min'
dat01_2_sa = pd.concat([dat01_sa[['AHU_CDU_SUM','AHU_FAN_SUM','1-122[℃]','1-122[%]']], tmp_1_sa['1-142[℃]_급기온도_10min']], axis=1)

## make output format per model
## delete NaN
dat01_2_sa = dat01_2_sa.dropna(axis=0)
in01_sa = dat01_2_sa.iloc[:, 0:-1]; out01_sa = dat01_2_sa.iloc[:, -1:]; 

## save scaler
scl01i_sa = MinMaxScaler()
scl01o_sa = MinMaxScaler()

scl01i_sa = scl01i_sa.fit(in01_sa)
scl01o_sa = scl01o_sa.fit(out01_sa)

joblib.dump(scl01i_sa, 'realIn_scaler.save')
joblib.dump(scl01o_sa, 'realOut_scaler.save')

in01_sa_scaled = scl01i_sa.transform(in01_sa)
out01_sa_scaled = scl01o_sa.transform(out01_sa)

## normalize the raw data
x01tr_sa, x01ts_sa, y01tr_sa, y01ts_sa = train_test_split(in01_sa_scaled, out01_sa_scaled, shuffle=False, test_size=0.2)
#%%
os.chdir(r'C:\Users\YourPCPC\Desktop\DS_MetaLr')
meta_ANN = build_ANN(out01_sa_scaled.shape[1], (in01_sa_scaled.shape[1],), (30, 30, 30))
#%%
checkpoint = ModelCheckpoint('./Model/saPred_bad_weight.h5', monitor = 'loss', verbose = 1, save_best_only = True)
#es = EarlyStopping(monitor='loss', mode='min', patience = 30)

meta_ANN.fit(x01tr_sa, y01tr_sa, epochs = 200, callbacks = [checkpoint])
#%%
meta_ANN.load_weights('./Model/saPred_bad_weight.h5')
meta_ANN.save('./Model/saPred_bad.h5')
#%%
prediction = scl01o_sa.inverse_transform(meta_ANN.predict(x01ts_sa))
true = scl01o_sa.inverse_transform(y01ts_sa)

print(cv_rmse(true, prediction))
print(mbe(true, prediction))
#%% plotting test result
parameters = {'axes.labelsize': 23,
          'axes.titlesize': 23,
          'xtick.labelsize': 23,
          'ytick.labelsize': 23,
          'font.family': 'Times New Roman'}
plt.rcParams.update(parameters)  

#%%
fig, ax1 = plt.subplots(1,1, figsize = (20, 5))

ax1.plot(true[0:144*10], color = 'b', label = 'measured')
ax1.plot(prediction[0:144*10], color = 'r', label = 'predicted')

ax1.set_xlabel('data points [10min]')
ax1.set_ylim([17.5, 27.5])
ax1.set_yticks(list(range(18,28,2)))
ax1.set_ylabel(u'temperature [\u00B0C]')

ax1.grid()
ax1.legend(loc = 'upper center', fontsize = 23, ncol = 2)

ax1.set_title('CVRMSE: {}%,       MBE: {}%'.format(np.round(cv_rmse(true,prediction), 2)[0],
                                                   np.round(mbe(true, prediction), 2)[0]))

fig.savefig('./Figure/badModel_test.png', dpi = 350, bbox_inches = 'tight')

#%%
test_input = scl01i_sa.inverse_transform(x01ts_sa)

fig, ax1 = plt.subplots(1,1, figsize = (20, 5))

ax1.plot(test_input[0:144*10, 0], color = 'g', label = 'CDU operation')

ax1.set_xlabel('data points [10min]')
ax1.set_ylim([-0.2, 5.2])
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_ylabel('no. of CDUs')

ax1.grid()
# ax1.legend(loc = 'upper center', fontsize = 23, ncol = 2)
ax2 = ax1.twinx()
ax2.plot(test_input[0:144*10, 1], color = 'magenta', label = 'fan on/off')

ax2.set_ylim([-0.2, 5.2])
ax2.set_yticks([0, 2])
ax2.set_yticklabels(['off', 'on'])
ax2.set_ylabel('on/off')

fig.legend(ncol = 2, bbox_to_anchor = (0.5, 0.88), fontsize = 23, loc = 'upper center')
fig.savefig('./Figure/badModel_input_operation.png', dpi = 350, bbox_inches = 'tight')