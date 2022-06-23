# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:35:58 2021

@author: Ra, S.J., Cho, S.
"""

#%% Import
import mlep
import mlep.mlep_decode_packet as decodePACKET
import mlep.mlep_encode_real_data as encodeDATA
import numpy as np
import socket
import os

import matplotlib.pyplot as plt
import sklearn.preprocessing as pre

from itertools import product
import random
import pandas as pd
import time
#%%
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def time_func(time_value):
    
    time_step = time_value/600 - 1.0
    dayofperiod = time_step//144
    timestepofday = time_step%144
    
    dayofweek = dayofperiod%7
    #hour = timestepofday//6
    
    return dayofweek, timestepofday

#%% make dir function
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
#%% change directory: not mandatory
os.chdir(r'C:\Users\YourPCPC\Desktop\IUES\workingfile\IUES_model_vs2\WORKING')
createFolder('C:/Users/YourPCPC/Desktop/IUES/workingfile/IUES_model_vs2/data_ann')


#%% co-simulation EnergyPlus via BCVTB
####MLEP initialize

MLEP_program = 'C:\\EnergyPlusV9-6-0\\RunEPlus.bat'     # EnergyPlus 설치 위치의 batch 파일. EP 버전 주의. 
MLEP_arguments = (r'C:\Users\Ggachicho93\Desktop\DS_MetaLr\DS_MetaLr_4coils_main', 'KOR_SO_Seoul.WS.471080_TMYx.2004-2018')    # 앞은 idf 파일 위치, 뒤는 %energyplus%/WeatherData 폴더의 epw 파일 이름
MLEP_accept_timeout = 20000     # Timeout for waiting for the client to connect
MLEP_port = 0                   # Socket port (default 0 = any free port)
MLEP_host = 'localhost'         # Host name (default '' = localhost)
MLEP_bcvtbDir = 'C:\\BCVTB'              # BCVTB 설치 위치 1 
MLEP_env = {'BCVTB_HOME': 'C:\\BCVTB'}   # BCVTB 설치 위치 2
MLEP_configFile = 'socket.cfg'      # socket 정의 
MLEP_configFileWriteOnce = False    # if true, only write the socket config file for the first time and when server socket changes
MLEP_exe_cmd = 'subprocess'
MLEP_workDir = r"C:\Users\Ggachicho93\Desktop\DS_MetaLr\WORKING"     # EP 시뮬레이션 수행할 임시 작업 폴더. 시뮬레이션 결과는 idf 파일 위치에 저장됨.

#%% Actions
action_space = [([0, 0, 0, 0],0),
                ([0, 0, 0, 0],1),
                ([1, 0, 0, 0],1),
                ([1, 1, 0, 0],1),
                ([1, 1, 1, 0],1),
                ([1, 1, 1, 1],1),]

dataMeta = list([])
#%%
for j in range(2):
    [MLEP_server_socket, MLEP_comm_socket, status, msg] = mlep.mlep_create(MLEP_program, MLEP_arguments, MLEP_workDir, MLEP_accept_timeout, MLEP_port, MLEP_host, MLEP_bcvtbDir, MLEP_configFile, MLEP_env, MLEP_exe_cmd)
    
    ## Socket connect
    # Accept Socket
    [MLEP_comm_socket, MLEP_client_address] = MLEP_server_socket.accept()
    # Create Streams
    if status == 0 and isinstance(MLEP_comm_socket, socket.socket):
        FLG_EP_is_running = True
        msg = ''
        print('Stream created!')
    #%% Dive into EP loop
    
    # Read outputs
    if FLG_EP_is_running:
        packet = MLEP_comm_socket.recv(5000)

    else:
        packet = ''
        print('Co-simulation is not running.')
    
    [FLG_EP_is_stopped, time_value, real_values] = decodePACKET(packet)
    
    # outputs (states)
    dataEP = np.asarray(real_values)
    
    timestepofday = 0
    dayofweek = 0
    
    previous_action = [0, 0, 0, 0]
    # Break EP simulation loop
    if FLG_EP_is_stopped == 1:
        print('EnergyPlus simulation ended. Python going to terminate the process.')
        FLG_EP_is_running = False
                
    #%%
    while FLG_EP_is_running == True:
        dataMeta.append(dataEP) 
        # Control variables (actions): Setpoint = 24, Heat Pump = On
        
        if time_value%1800 == 0.0:
            action_no = random.randint(0,len(action_space)-1)
            action = action_space[action_no]
            
        else:
            action = previous_action
        
        ### [CDU1, CDU2, CDU3, CDU4, Fan]
        dataPy = [action[0][0], action[0][1], action[0][2], action[0][3], action[1]]
        
    
        ## WRITE EnergyPlus
        # ENCODE Packet
        tuplePy = totuple(dataPy)
        packet = encodeDATA(2, FLG_EP_is_stopped, time_value, tuplePy)
        
        # Write Packet
        if FLG_EP_is_running:
            packet = packet.encode(encoding='UTF-8')
            MLEP_comm_socket.sendall(packet)
        else:
            print('Co-simulation is not running.')
        
        
        #Read
        if FLG_EP_is_running:
            packet = MLEP_comm_socket.recv(5000)
        else:
            packet = ''
            print('Co-simulation is not running.')
    
        [FLG_EP_is_stopped, time_value, real_values] = decodePACKET(packet)
        
        # outputs (states)
        dataEP = np.asarray(real_values)
        
        if FLG_EP_is_stopped == 1:
        
            FLG_EP_is_running = False
            break
        
        dayofweek, timestepofday = time_func(time_value)
        previous_action = action
    #%% Stop EnergyPlus
    
    # ENCODE last packet
    tuplePy = totuple([0,0])
    packet = encodeDATA(2, FLG_EP_is_stopped, time_value, tuplePy)
    
    # Write last packet
    packet = packet.encode(encoding='UTF-8')
    MLEP_comm_socket.sendall(packet)
    
    # Close connection
    MLEP_comm_socket.close()  
    MLEP_comm_socket = None
    time.sleep(2)
#%% save the synthetic data
columns = ['OAT', 'OAH', 'CDU1', 'CDU2', 'CDU3', 'CDU4', 'Fan', 'SAT']
Meta_df = pd.DataFrame(dataMeta)
Meta_df.columns = columns
Meta_df.to_csv(r'C:\Users\YourPCPC\Desktop\DS_MetaLr\dataMeta_vs2.csv', index = False)
