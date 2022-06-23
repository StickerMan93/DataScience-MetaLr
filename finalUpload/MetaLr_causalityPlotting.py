# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:44:56 2022

@author: Ra, S.J., Cho, S.
"""

import pandas as pd
import numpy as np
import sklearn
import os

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model


os.chdir(r'C:\Users\YourPCPC\Desktop\DS_MetaLr')
#%% Metrics
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
        
#%% Load scalers
scl01i_sa = joblib.load('./Model/realIn_scaler.save')
scl01o_sa = joblib.load('./Model/realOut_scaler.save')

#%% pre-process data with experiments: [0,1,2,3,4]
os.chdir('./ANN_model')
## read xlsx
z2temp = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_meta.xlsx",'Doosan_Testo_2105-06', engine='openpyxl', index_col='Time')
z2temp_sa = pd.read_excel(r"Doosan_TESTO_2105-06_vs2_valid_meta.xlsx",'Sheet1', engine='openpyxl', index_col='Time')

# AHU 가동상태
z2ctrl = pd.read_excel(r"Doosan_AHU_2105-06_meta_vs2.xlsx",'origin', engine='openpyxl', index_col='Time')
Z2ctrl_bad = pd.read_excel(r"Doosan_AHU_2105-06_meta_vs2.xlsx",'on_off', engine='openpyxl', index_col='Time')
z2temp.index = z2temp.index.round('min')
z2temp_sa.index = z2temp_sa.index.round('min')

# retime
z2temp=z2temp.loc["2021-07-30":"2021-09-17",:]
z2ctrl=z2ctrl.loc["2021-07-30":"2021-09-17",:]
z2temp_sa=z2temp_sa.loc["2021-07-30":"2021-09-17",:]
Z2ctrl_bad = Z2ctrl_bad.loc["2021-07-30":"2021-09-17",:]

####
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
in01_sa_scaled_vs2 = scl01i_sa.transform(in01_sa)
out01_sa_scaled_vs2 = scl01o_sa.transform(out01_sa)

# ## divide data (20%: test)
x01tr_sa, x01ts_sa, y01tr_sa, y01ts_sa = train_test_split(in01_sa_scaled_vs2, out01_sa_scaled_vs2, shuffle=False, test_size=0.2)

os.chdir(r'C:\Users\YourPCPC\Desktop\DS_MetaLr')
#%% plotting test result
parameters = {'axes.labelsize': 23,
          'axes.titlesize': 23,
          'xtick.labelsize': 23,
          'ytick.labelsize': 23,
          'font.family': 'Times New Roman'}
plt.rcParams.update(parameters)
#%% Causality Plotting
### measured data
true = scl01o_sa.inverse_transform(y01ts_sa)
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)
data_arr = np.append(input_unscaled, true, axis = 1)
data_df_true = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_true['deltaT'] = data_df_true['OAT'] - data_df_true['SAT']
data_df_true['CDU'] = np.round(data_df_true['CDU'])

data_AHU0 = data_df_true[data_df_true['CDU'] == 0]
data_AHU1 = data_df_true[data_df_true['CDU'] == 1]
data_AHU2 = data_df_true[data_df_true['CDU'] == 2]
data_AHU3 = data_df_true[data_df_true['CDU'] == 3]
data_AHU4 = data_df_true[data_df_true['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/real data causality.png', dpi = 350, bbox_inches = 'tight')

#%% model 1 prediction causality

bad_ANN = load_model('./Model/saPred_bad.h5')
prediction = scl01o_sa.inverse_transform(bad_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_bad = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_bad['deltaT'] = data_df_bad['OAT'] - data_df_bad['SAT']
data_df_bad['CDU'] = np.round(data_df_bad['CDU'])

data_AHU0 = data_df_bad[data_df_bad['CDU'] == 0]
data_AHU1 = data_df_bad[data_df_bad['CDU'] == 1]
data_AHU2 = data_df_bad[data_df_bad['CDU'] == 2]
data_AHU3 = data_df_bad[data_df_bad['CDU'] == 3]
data_AHU4 = data_df_bad[data_df_bad['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/bad model causality.png', dpi = 350, bbox_inches = 'tight')

#%% model 6 prediction causality
trans_ANN = load_model('./Model/saPred_Transferred.h5')
prediction = scl01o_sa.inverse_transform(trans_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_trans = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_trans['deltaT'] = data_df_trans['OAT'] - data_df_trans['SAT']
data_df_trans['CDU'] = np.round(data_df_trans['CDU'])

data_AHU0 = data_df_trans[data_df_trans['CDU'] == 0]
data_AHU1 = data_df_trans[data_df_trans['CDU'] == 1]
data_AHU2 = data_df_trans[data_df_trans['CDU'] == 2]
data_AHU3 = data_df_trans[data_df_trans['CDU'] == 3]
data_AHU4 = data_df_trans[data_df_trans['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/trans model causality.png', dpi = 350, bbox_inches = 'tight')

#%% ep model prediction causality
ep_ANN = load_model('./Model/saPred_EP.h5')
prediction = scl01o_sa.inverse_transform(ep_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_ep = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_ep['deltaT'] = data_df_ep['OAT'] - data_df_ep['SAT']
data_df_ep['CDU'] = np.round(data_df_ep['CDU'])

data_AHU0 = data_df_ep[data_df_ep['CDU'] == 0]
data_AHU1 = data_df_ep[data_df_ep['CDU'] == 1]
data_AHU2 = data_df_ep[data_df_ep['CDU'] == 2]
data_AHU3 = data_df_ep[data_df_ep['CDU'] == 3]
data_AHU4 = data_df_ep[data_df_ep['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/ep model causality.png', dpi = 350, bbox_inches = 'tight')

#%% target model prediction causality
target_ANN = load_model('./Model/saPred_Target.h5')
prediction = scl01o_sa.inverse_transform(target_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_target = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_target['deltaT'] = data_df_target['OAT'] - data_df_target['SAT']
data_df_target['CDU'] = np.round(data_df_target['CDU'])

data_AHU0 = data_df_target[data_df_target['CDU'] == 0]
data_AHU1 = data_df_target[data_df_target['CDU'] == 1]
data_AHU2 = data_df_target[data_df_target['CDU'] == 2]
data_AHU3 = data_df_target[data_df_target['CDU'] == 3]
data_AHU4 = data_df_target[data_df_target['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/target model causality.png', dpi = 350, bbox_inches = 'tight')

#%% model 5 prediction causality
ep_ANN = load_model('./Model/saPred_EP.h5')
bad_ANN = load_model('./Model/saPred_bad.h5')

trans_weights = bad_ANN.layers[3].get_weights()
ep_ANN.layers[3].set_weights(trans_weights)

prediction = scl01o_sa.inverse_transform(ep_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_case1 = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_case1['deltaT'] = data_df_case1['OAT'] - data_df_case1['SAT']
data_df_case1['CDU'] = np.round(data_df_case1['CDU'])

data_AHU0 = data_df_case1[data_df_case1['CDU'] == 0]
data_AHU1 = data_df_case1[data_df_case1['CDU'] == 1]
data_AHU2 = data_df_case1[data_df_case1['CDU'] == 2]
data_AHU3 = data_df_case1[data_df_case1['CDU'] == 3]
data_AHU4 = data_df_case1[data_df_case1['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/case1 model causality.png', dpi = 350, bbox_inches = 'tight')

#%% model 4 prediction causality
ep_ANN = load_model('./Model/saPred_EP.h5')
bad_ANN = load_model('./Model/saPred_bad.h5')

trans_weights = ep_ANN.layers[3].get_weights()
bad_ANN.layers[3].set_weights(trans_weights)

prediction = scl01o_sa.inverse_transform(bad_ANN.predict(x01ts_sa))
input_unscaled = scl01i_sa.inverse_transform(x01ts_sa)

data_arr = np.append(input_unscaled, prediction, axis = 1)
data_df_case2 = pd.DataFrame(data_arr, columns = ['CDU', 'Fan', 'OAT', 'OARH', 'SAT'])
data_df_case2['deltaT'] = data_df_case2['OAT'] - data_df_case2['SAT']
data_df_case2['CDU'] = np.round(data_df_case2['CDU'])

data_AHU0 = data_df_case2[data_df_case2['CDU'] == 0]
data_AHU1 = data_df_case2[data_df_case2['CDU'] == 1]
data_AHU2 = data_df_case2[data_df_case2['CDU'] == 2]
data_AHU3 = data_df_case2[data_df_case2['CDU'] == 3]
data_AHU4 = data_df_case2[data_df_case2['CDU'] == 4]

###
fig, ax1 = plt.subplots(1,1, figsize = (12,5))

sns.distplot(data_AHU0['deltaT'], hist = False, label = 'CDU 0',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU1['deltaT'], hist = False, label = 'CDU 1',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU2['deltaT'], hist = False, label = 'CDU 2',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU3['deltaT'], hist = False, label = 'CDU 3',
             kde_kws={'linewidth': 2, 'shade':True})
sns.distplot(data_AHU4['deltaT'], hist = False, label = 'CDU 4',
             kde_kws={'linewidth': 2, 'shade':True})

ax1.set_xlim([-6, 13])
ax1.set_xticks([-5, 0, 5, 10])
ax1.set_xlabel(u'delta T [\u00B0C]')

ax1.set_ylim([-0.05, 1.55])
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
ax1.set_ylabel('density')
ax1.grid()
ax1.legend(loc = 'upper right', fontsize = 23, ncol = 1)

fig.savefig('./Figure/causality/case2 model causality.png', dpi = 350, bbox_inches = 'tight')