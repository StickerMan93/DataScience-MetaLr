# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:51:54 2022

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

createFolder('./Model')
#%%
virtual_Data = pd.read_csv('./dataMeta_vs2.csv', header = [0])
virtual_Data['CDU_oper'] = virtual_Data[['CDU1', 'CDU2', 'CDU3', 'CDU4']].sum(axis = 1)
virtual_Data['Fan'] = virtual_Data['Fan']*2

inputData = virtual_Data[['CDU_oper', 'Fan', 'OAT', 'OAH']]
outputData = virtual_Data[['SAT']]

### Make scalers
inputScaler = MinMaxScaler()
outputScaler = MinMaxScaler()

inputScaler = inputScaler.fit(inputData)
outputScaler = outputScaler.fit(outputData)

joblib.dump(inputScaler, './Model/input_scaler.save')
joblib.dump(outputScaler, './Model/output_scaler.save')

### Scale input/output data
inputScaled = inputScaler.transform(inputData)
outputScaled = outputScaler.transform(outputData)

###
inputTrain = inputScaled
outputTrain = outputScaled

#%%
train_input, test_input, train_output, test_output = train_test_split(inputTrain, outputTrain, test_size = 0.3, shuffle = False, random_state = 12)
meta_ANN = build_ANN(outputTrain.shape[1], (inputTrain.shape[1],), (30, 30, 30))

#%%
checkpoint = ModelCheckpoint('./Model/saPred_EP_weight.h5', monitor = 'loss', verbose = 1, save_best_only = True)
#es = EarlyStopping(monitor='loss', mode='min', patience = 30)

meta_ANN.fit(train_input, train_output, epochs = 200, callbacks = [checkpoint])
#%%
meta_ANN.load_weights('./Model/saPred_EP_weight.h5')
meta_ANN.save('./Model/saPred_EP.h5')

#%%
prediction = outputScaler.inverse_transform(meta_ANN.predict(test_input))
true = outputScaler.inverse_transform(test_output)
print(cv_rmse(true, prediction))
#%%
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
ax1.set_ylim([12.5, 35.5])
ax1.set_yticks(list(range(13,35,5)))
ax1.set_ylabel(u'temperature [\u00B0C]')

ax1.grid()
ax1.legend(loc = 'upper center', fontsize = 23, ncol = 2)

ax1.set_title('CVRMSE: {}%,       MBE: {}%'.format(np.round(cv_rmse(true,prediction), 2)[0],
                                                   np.round(mbe(true, prediction), 2)[0]))

fig.savefig('./Figure/epModel_test.png', dpi = 350, bbox_inches = 'tight')

#%%
test_input_unscaled = inputScaler.inverse_transform(test_input)

fig, ax1 = plt.subplots(1,1, figsize = (20, 5))

ax1.plot(test_input_unscaled[0:144*10, 0], color = 'g', label = 'CDU operation')

ax1.set_xlabel('data points [10min]')
ax1.set_ylim([-0.2, 5.2])
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_ylabel('no. of CDUs')

ax1.grid()
# ax1.legend(loc = 'upper center', fontsize = 23, ncol = 2)
ax2 = ax1.twinx()
ax2.plot(test_input_unscaled[0:144*10, 1], color = 'magenta', label = 'fan on/off')

ax2.set_ylim([-0.2, 5.2])
ax2.set_yticks([0, 2])
ax2.set_yticklabels(['off', 'on'])
ax2.set_ylabel('on/off')
 
fig.legend(ncol = 2, bbox_to_anchor = (0.5, 0.88), fontsize = 23, loc = 'upper center')
fig.savefig('./Figure/epModel_input_operation.png', dpi = 350, bbox_inches = 'tight')