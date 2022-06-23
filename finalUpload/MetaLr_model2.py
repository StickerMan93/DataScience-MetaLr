# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:07:05 2022

@author: Ggachicho93
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

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.initializers import RandomNormal, Zeros

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


os.chdir(r'C:\Users\Ggachicho93\Desktop\DS_MetaLr')
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

#%% Load models and scalers
## Load models
ep_ANN = load_model('./Model/saPred_EP.h5')
# bad_ANN = load_model('./Model/saPred_bad.h5')

## Load scalers
scl01i_sa = joblib.load('./ANN_model/realIn_scaler.save')
scl01o_sa = joblib.load('./ANN_model/realOut_scaler.save')

#%%
os.chdir('./ANN_model')

"1. 데이터 불러오기"
# data pre-process
## read xlsx
# 실내/외기온도
# z2temp = pd.read_excel(r"Doosan_KETI_6789.xlsx",'whole', engine='openpyxl', index_col='Time')
z2temp = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_meta.xlsx",'Doosan_Testo_2105-06', engine='openpyxl', index_col='Time')
z2temp_sa = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_valid_meta.xlsx",'Sheet1', engine='openpyxl', index_col='Time')

# AHU 가동상태
#z2ctrl = pd.read_excel(r"Doosan_AHU_6789_vs2.xlsx",'whole_6789', engine='openpyxl', index_col='Time')
z2ctrl = pd.read_excel(r"Doosan_AHU_2105-06_meta_vs2.xlsx",'origin', engine='openpyxl', index_col='Time')
Z2ctrl_bad = pd.read_excel(r"Doosan_AHU_2105-06_meta_vs2.xlsx",'on_off', engine='openpyxl', index_col='Time')
z2temp.index = z2temp.index.round('min')
#z2ctrl.index = z2ctrl.index.round('min')
z2temp_sa.index = z2temp_sa.index.round('min')
# retime
z2temp=z2temp.loc["2021-07-30":"2021-09-17",:]
z2ctrl=z2ctrl.loc["2021-07-30":"2021-09-17",:]
z2temp_sa=z2temp_sa.loc["2021-07-30":"2021-09-17",:]
Z2ctrl_bad = Z2ctrl_bad.loc["2021-07-30":"2021-09-17",:]
#%%
## make input format per model
# 3구역: 1차 실증 구역 #1
dat01_sa = pd.concat([z2ctrl[['AHU_CDU_SUM','AHU_FAN_SUM']],z2temp[['1-122[℃]','1-122[%]','1-142[℃]_급기온도']]], axis=1)  # z3: 1차 실증 구역 #1
dat01_sa = dat01_sa.dropna(axis=0)
## accumulate over time: (t) -> (t, t-10min)
# 10분전 데이터
tmp_1_sa = dat01_sa.shift(-10)
tmp_1_sa.columns = tmp_1_sa.columns + '_10min'

# 현재 데이터 + 10분전 데이터 결합
dat01_2_sa = pd.concat([dat01_sa[['AHU_CDU_SUM','AHU_FAN_SUM','1-122[℃]','1-122[%]']], tmp_1_sa['1-142[℃]_급기온도_10min']], axis=1)

## make output format per model
## delete NaN
dat01_2_sa = dat01_2_sa.dropna(axis=0)
in01_sa = dat01_2_sa.iloc[:, 0:-1]; out01_sa = dat01_2_sa.iloc[:, -1:]; 

## save scaler
in01_sa_scaled_vs2 = scl01i_sa.transform(in01_sa)
out01_sa_scaled_vs2 = scl01o_sa.transform(out01_sa)

# ## divide data (20%: test)
x01tr_sa, x01ts_sa, y01tr_sa, y01ts_sa = train_test_split(in01_sa_scaled_vs2, out01_sa_scaled_vs2, shuffle=False, test_size=0.2)

#%%
prediction = scl01o_sa.inverse_transform(ep_ANN.predict(x01ts_sa))
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
os.chdir(r'C:\Users\Ggachicho93\Desktop\DS_MetaLr')
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

fig.savefig('./Figure/epModel_test.png', dpi = 350, bbox_inches = 'tight')

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
fig.savefig('./Figure/epModel_input_operation.png', dpi = 350, bbox_inches = 'tight')