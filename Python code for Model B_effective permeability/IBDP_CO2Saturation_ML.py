# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:00:09 2023

@author: xzhang
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') # The magic('reset -sf') statement is used to
#clear all variables before the script begins to run.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import load
from pickle import dump

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
scaler = MinMaxScaler()

import gc

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import datetime
ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
    
# path = os.getcwd() #stands for "get current working directory"
# print(path)
os.chdir('C:/Users/xzhang/Downloads/train/') #Change the current working directory
#%% simulation information
trainCases = 72
testCases = 10
model_xsize = 32
model_ysize = 32
model_zsize = 32
n_months = 50 #(monthly output) 

root_directory = 'C:/Users/xzhang/Desktop/train/dataset'

x_coordinates = np.load(f'{root_directory}/x_coordinates.npy').astype('float32')
y_coordinates = np.load(f'{root_directory}/y_coordinates.npy').astype('float32')
z_coordinates = np.load(f'{root_directory}/z_coordinates.npy').astype('float32')

perm_xyz_train = np.load(f'{root_directory}/perm_xyz_train.npy').astype('float32')
porosity_train = np.load(f'{root_directory}/porosity_train.npy').astype('float32')
inject_gir_month_train = np.load(f'{root_directory}/inject_gir_month_train.npy').astype('float32')
time_train = np.load(f'{root_directory}/time_train.npy').astype('float32')
transMulti_xyz_train = np.load(f'{root_directory}/transMulti_xyz_train.npy').astype('float32')
capi_drainage_train_orig = np.zeros([trainCases,model_xsize,model_ysize,model_zsize,n_months])
rePerm_drainage_train_orig = np.zeros([trainCases,model_xsize,model_ysize,model_zsize,n_months])
drainage_satTable_trainOnly = np.load(f'{root_directory}/drainage_satTable_trainOnly.npy')
drainage_satTable_validate = np.load(f'{root_directory}/drainage_satTable_validate.npy')

saturation_dataset_train = np.load(f'{root_directory}/saturation_dataset_train.npy').astype('float32')
drainage_satMax_train = np.load(f'{root_directory}/drainage_satMax_train.npy').astype('float32')

##############################################################################################
perm_xyz_test = np.load(f'{root_directory}/perm_xyz_test.npy').astype('float32')
porosity_test = np.load(f'{root_directory}/porosity_test.npy').astype('float32')
inject_gir_month_test = np.load(f'{root_directory}/inject_gir_month_test.npy').astype('float32')
time_test = np.load(f'{root_directory}/time_test.npy').astype('float32')
transMulti_xyz_test = np.load(f'{root_directory}/transMulti_xyz_test.npy').astype('float32')
drainage_satTable_test = np.load(f'{root_directory}/drainage_satTable_test.npy')

saturation_dataset_test = np.load(f'{root_directory}/saturation_dataset_test.npy').astype('float32')
drainage_satMax_test = np.load(f'{root_directory}/drainage_satMax_test.npy').astype('float32')

perm_xyz_max = 944.5927
perm_xyz_min = 5.34701e-05

porosity_max = 0.274972
porosity_min = 0.000115415
# =============================================================================

#######################################################################################
from DfpNet_3D import TurbNetG, weights_init

lambda_ = 0.0001
        
lrG = 0.0001 # learning rate
currLr = lrG

print("LR: {}".format(lrG))
# decay learning rate?
decayLr = True
print("LR decay: {}".format(decayLr))

# channel exponent to control network size
expo = 4

saveL1 = False 

prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))
    
dropout    = 0.
doLoad     = ""      # optional, path to pre-trained model
netG = TurbNetG(channelExponent=expo, dropout=dropout)
print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
    
criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.)

trainCases = np.size(perm_xyz_train,0)

data_ML_validate_index = np.arange(8, trainCases, 4)
data_ML_train_index = np.delete(np.arange(0,trainCases),data_ML_validate_index)

validateCases = data_ML_validate_index.shape[0]

n_months_train = n_months-1
train_cases = trainCases-validateCases
all_train_size = n_months_train*trainCases
validate_size = n_months_train*validateCases
train_size = n_months_train*train_cases

input_channels = 10

X_train = np.zeros([train_size, model_xsize, model_ysize, model_zsize, input_channels])
X_validate = np.zeros([validate_size, model_xsize, model_ysize, model_zsize, input_channels])

y_train = np.zeros([train_size, model_xsize, model_ysize, model_zsize])
y_validate = np.zeros([validate_size, model_xsize, model_ysize, model_zsize]) 

y_drainage_satMax_train = np.zeros([train_size, model_xsize, model_ysize, model_zsize])
y_drainage_satMax_validate = np.zeros([validate_size, model_xsize, model_ysize, model_zsize])
for i in range(train_cases):
    for month in range(n_months_train):   
        y_train[i*n_months_train+month,:,:,:] = saturation_dataset_train[data_ML_train_index[i],:,:,:,month+1]
        
        y_drainage_satMax_train[i*n_months_train+month,:,:,:] = drainage_satMax_train[data_ML_train_index[i],:,:,:]
        
        X_train[i*n_months_train+month,:,:,:,0] = perm_xyz_train[data_ML_train_index[i],:,:,:,0]*\
        rePerm_drainage_train_orig[data_ML_train_index[i],:,:,:,month]
        X_train[i*n_months_train+month,:,:,:,1] = perm_xyz_train[data_ML_train_index[i],:,:,:,1]*\
        rePerm_drainage_train_orig[data_ML_train_index[i],:,:,:,month]
        X_train[i*n_months_train+month,:,:,:,2] = perm_xyz_train[data_ML_train_index[i],:,:,:,2]*\
        rePerm_drainage_train_orig[data_ML_train_index[i],:,:,:,month]
        
        X_train[i*n_months_train+month,:,:,:,3] = porosity_train[data_ML_train_index[i],:,:,:]
        X_train[i*n_months_train+month,:,:,:,4] = inject_gir_month_train[data_ML_train_index[i],:,:,:,month]
        
        X_train[i*n_months_train+month,:,:,:,5] = time_train[data_ML_train_index[i],:,:,:,month+1] 
        
        X_train[i*n_months_train+month,:,:,:,6:9] = transMulti_xyz_train[data_ML_train_index[i],:,:,:,:]
        
        X_train[i*n_months_train+month,:,:,:,9] = capi_drainage_train_orig[data_ML_train_index[i],:,:,:,month]
        
for i in range(validateCases):
    for month in range(n_months_train):       
        y_validate[i*n_months_train+month,:,:,:] = saturation_dataset_train[data_ML_validate_index[i],:,:,:,month+1]
        
        y_drainage_satMax_validate[i*n_months_train+month,:,:,:] = drainage_satMax_train[data_ML_validate_index[i],:,:,:]
        
        X_validate[i*n_months_train+month,:,:,:,0] = perm_xyz_train[data_ML_validate_index[i],:,:,:,0]*\
        rePerm_drainage_train_orig[data_ML_validate_index[i],:,:,:,month]    
        X_validate[i*n_months_train+month,:,:,:,1] = perm_xyz_train[data_ML_validate_index[i],:,:,:,1]*\
        rePerm_drainage_train_orig[data_ML_validate_index[i],:,:,:,month]    
        X_validate[i*n_months_train+month,:,:,:,2] = perm_xyz_train[data_ML_validate_index[i],:,:,:,2]*\
        rePerm_drainage_train_orig[data_ML_validate_index[i],:,:,:,month]  
        
        X_validate[i*n_months_train+month,:,:,:,3] = porosity_train[data_ML_validate_index[i],:,:,:]
        X_validate[i*n_months_train+month,:,:,:,4] = inject_gir_month_train[data_ML_validate_index[i],:,:,:,month]
        
        X_validate[i*n_months_train+month,:,:,:,5] = time_train[data_ML_validate_index[i],:,:,:,month+1] 
        
        X_validate[i*n_months_train+month,:,:,:,6:9] = transMulti_xyz_train[data_ML_validate_index[i],:,:,:,:]
        
        X_validate[i*n_months_train+month,:,:,:,9] = capi_drainage_train_orig[data_ML_validate_index[i],:,:,:,month]
        
X_train_roll = np.rollaxis(X_train,4,1)
X_validate_roll = np.rollaxis(X_validate,4,1)

# batch size
batchSizeCoeffi = 4
batch_size_train = n_months_train*batchSizeCoeffi
batch_size_validate = n_months_train*batchSizeCoeffi
    
targets_train = Variable(torch.FloatTensor(batch_size_train, model_xsize, model_ysize, model_zsize))
inputs_train  = Variable(torch.FloatTensor(batch_size_train, input_channels, model_xsize, model_ysize, model_zsize))
    
targets_validate = Variable(torch.FloatTensor(batch_size_validate, model_xsize, model_ysize, model_zsize))
inputs_validate  = Variable(torch.FloatTensor(batch_size_validate, input_channels, model_xsize, model_ysize, model_zsize))
    
train_batch_number = int(np.size(X_train_roll,0)/batch_size_train)
X_train_batch = X_train_roll.reshape((train_batch_number, batch_size_train, X_train_roll.shape[1], X_train_roll.shape[2],\
                                     X_train_roll.shape[3], X_train_roll.shape[4]))
y_train_batch = y_train.reshape((train_batch_number, batch_size_train, y_train.shape[1], y_train.shape[2], y_train.shape[3]))

y_drainage_satMax_train_batch = y_drainage_satMax_train.reshape((train_batch_number, batch_size_train, y_train.shape[1], y_train.shape[2], y_train.shape[3]))
    
validate_batch_number = int(np.size(X_validate_roll,0)/batch_size_validate)
X_validate_batch = X_validate_roll.reshape((validate_batch_number, batch_size_validate, X_validate_roll.shape[1], \
                                           X_validate_roll.shape[2], X_validate_roll.shape[3], X_validate_roll.shape[4]))
y_validate_batch = y_validate.reshape((validate_batch_number, batch_size_validate, y_validate.shape[1], \
                                           y_validate.shape[2], y_validate.shape[3]))    
y_drainage_satMax_validate_batch = y_drainage_satMax_validate.reshape((validate_batch_number, batch_size_validate, y_validate.shape[1], y_validate.shape[2], y_validate.shape[3]))
    
# setup training
epochs = 800
epochs_drop = epochs/10

L1_accum_array = np.zeros(epochs)
L1val_accum_array = np.zeros(epochs)   

from satFuncFitting import satFuncFitting
zero_array_temp = torch.tensor(np.zeros([batch_size_train, y_train.shape[1], y_train.shape[2], y_train.shape[3]]))

f_trainError = open("./error_recordFiles/trainError.txt", "w")  
f_validateError = open("./error_recordFiles/validateError.txt", "w")  

f_trainErrorOnly = open("./error_recordFiles/trainErrorOnly.txt", "w")  

f_trainError_batchAverage = open("./error_recordFiles/batchAverage_trainError.txt", "w")  
f_validateError_batchAverage = open("./error_recordFiles/batchAverage_validateError.txt", "w")  

y_predict_train = np.zeros([train_cases, model_xsize, model_ysize, model_zsize, n_months])
y_predict_validate = np.zeros([validateCases, model_xsize, model_ysize, model_zsize, n_months])

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))
       
    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(X_train_batch, 0):
        #print('i = ', i)
        # traindata
        #inputs_cpu, targets_cpu = traindata
        traindata = torch.tensor(traindata)
        targetdata = torch.tensor(y_train_batch[i,:,:,:,:])
        inputs_train.data.copy_(traindata.float())
        targets_train.data.copy_(targetdata.float())                  
                      
        netG.zero_grad()
        gen_out = netG(inputs_train)            
        gen_out = torch.squeeze(gen_out,1)   
        gen_out_array = gen_out.data.numpy()   
        
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_train[batchSizeCoeffi*i+bc,:,:,:,month+1] = gen_out_array[month+bc*n_months_train,:,:,:]

        lossL1 = criterionL1(gen_out, targets_train)
        lossL1.backward()
        optimizerG.step()
                   
        lossL1train = lossL1.item()
        L1_accum += lossL1.item()
    
        if i==len(X_train_batch)-1:
            
            logline = "Epoch: {}, batch-idx: {}, currLr: {}\n".format(epoch, i, currLr)
            print(logline)
            
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1train)
            print(logline)            
            
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            f_trainError.write(theTime+'\n')
            f_trainError.flush()
            f_trainError.write('Epoch:' +str(epoch)+', batch-id:' +str(i)+', Learning rate: '+str(currLr)+'\n')
            f_trainError.flush()
            f_trainError.write('total: '+str(lossL1train)+'\n')
            f_trainError.flush()
            f_trainError.write('L1: '+str(lossL1.item())+'\n')
            f_trainError.flush()
            
            f_trainErrorOnly.write(theTime+'\n')
            f_trainErrorOnly.flush()
            f_trainErrorOnly.write('L1: '+str(lossL1.item())+'\n')
            f_trainErrorOnly.flush()
            
            torch.save(netG, './saved_trainedModels/netG_model.pt')
            
    rePermCapi_drainage_train = satFuncFitting(y_predict_train,drainage_satTable_trainOnly)
    for index in range(train_batch_number):
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):  
                X_train_batch[index,month+bc*n_months_train,0,:,:,:] = perm_xyz_train[data_ML_train_index[batchSizeCoeffi*index+bc],:,:,:,0]*\
                    rePermCapi_drainage_train[month][0][batchSizeCoeffi*i+bc,:,:,:]
                X_train_batch[index,month+bc*n_months_train,1,:,:,:] = perm_xyz_train[data_ML_train_index[batchSizeCoeffi*index+bc],:,:,:,1]*\
                    rePermCapi_drainage_train[month][0][batchSizeCoeffi*i+bc,:,:,:]
                X_train_batch[index,month+bc*n_months_train,2,:,:,:] = perm_xyz_train[data_ML_train_index[batchSizeCoeffi*index+bc],:,:,:,2]*\
                    rePermCapi_drainage_train[month][0][batchSizeCoeffi*i+bc,:,:,:]
                
                X_train_batch[index,month+bc*n_months_train,9,:,:,:] = rePermCapi_drainage_train[month][1][batchSizeCoeffi*index+bc,:,:,:]   

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(X_validate_batch, 0):
        #inputs_cpu, targets_cpu = validata
        validata = torch.tensor(validata)
        targetdata = torch.tensor(y_validate_batch[i,:,:,:,:])
        inputs_validate.data.copy_(validata.float())
        targets_validate.data.copy_(targetdata.float())
    
        outputs = netG(inputs_validate)            
        outputs = torch.squeeze(outputs,1)            
        outputs_cpu = outputs.data.cpu().numpy()            
        targets_cpu = targets_validate.data.cpu().numpy()            
        outputs_array = outputs.data.numpy() 
        
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_validate[batchSizeCoeffi*i+bc,:,:,:,month+1] = outputs_array[month+bc*n_months_train,:,:,:]
            
        lossL1 = criterionL1(outputs, targets_validate)
        L1val_accum += lossL1.item()     
        
        if i==len(X_validate_batch)-1:
            logline = "Epoch: {}, batch-idx: {}, L1 validation: {}\n".format(epoch, i, lossL1.item())
            print(logline)
                        
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            f_validateError.write(theTime+'\n')
            f_validateError.flush()
            f_validateError.write('Epoch:' +str(epoch)+', batch-id:' +str(i)+', validation error: '+str(lossL1.item())+'\n')
            f_validateError.flush()
            
    rePermCapi_drainage_validate = satFuncFitting(y_predict_validate,drainage_satTable_validate)
    for index in range(validate_batch_number):
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):  
                X_validate_batch[index,month+bc*n_months_train,0,:,:,:] = perm_xyz_train[data_ML_validate_index[batchSizeCoeffi*index+bc],:,:,:,0]*\
                    rePermCapi_drainage_validate[month][0][batchSizeCoeffi*i+bc,:,:,:]
                X_validate_batch[index,month+bc*n_months_train,1,:,:,:] = perm_xyz_train[data_ML_validate_index[batchSizeCoeffi*index+bc],:,:,:,1]*\
                    rePermCapi_drainage_validate[month][0][batchSizeCoeffi*i+bc,:,:,:]
                X_validate_batch[index,month+bc*n_months_train,2,:,:,:] = perm_xyz_train[data_ML_validate_index[batchSizeCoeffi*index+bc],:,:,:,2]*\
                    rePermCapi_drainage_validate[month][0][batchSizeCoeffi*i+bc,:,:,:]
                
                X_validate_batch[index,month+bc*n_months_train,9,:,:,:] = rePermCapi_drainage_validate[month][1][batchSizeCoeffi*index+bc,:,:,:]   
    
    # data for graph plotting
    L1_accum    /= len(X_train_batch)
    L1val_accum /= len(X_validate_batch)    
    L1_accum_array[epoch] = L1_accum
    L1val_accum_array[epoch] = L1val_accum 
    
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    f_trainError_batchAverage.write(theTime+'\n')
    f_trainError_batchAverage.flush() 
    f_trainError_batchAverage.write('Epoch:' +str(epoch)+', average train error: '+str(L1_accum)+'\n') 
    f_trainError_batchAverage.flush()      
    
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    f_validateError_batchAverage.write(theTime+'\n')
    f_validateError_batchAverage.flush()
    f_validateError_batchAverage.write('Epoch:' +str(epoch)+', average validation error: '+str(L1val_accum)+'\n')   
    f_validateError_batchAverage.flush()            
    
f_trainError.close()        
f_validateError.close()
f_trainErrorOnly.close()        
f_trainError_batchAverage.close() 
f_validateError_batchAverage.close()

# test       
netG_model = torch.load('./saved_trainedModels/netG_model.pt')

# netG.eval()
netG_model.eval()

testCase_interval = 16
test_cases = perm_xyz_test.shape[0]
test_size = n_months*test_cases

X_test = np.zeros([test_cases, model_xsize, model_ysize, model_zsize, input_channels])

X_test[:,:,:,:,3] = porosity_test    

X_test[:,:,:,:,6:9] = transMulti_xyz_test

x_permeability_recovered_array = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                             saturation_dataset_test.shape[2],saturation_dataset_test.shape[3]])
y_permeability_recovered_array = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                             saturation_dataset_test.shape[2],saturation_dataset_test.shape[3]])
z_permeability_recovered_array = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                             saturation_dataset_test.shape[2],saturation_dataset_test.shape[3]])
porosity_recovered_array = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                             saturation_dataset_test.shape[2],saturation_dataset_test.shape[3]])

saturation_test_recovered = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                             saturation_dataset_test.shape[2],saturation_dataset_test.shape[3],\
                             saturation_dataset_test.shape[4]])
saturation_predict_recovered = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                                       saturation_dataset_test.shape[2],saturation_dataset_test.shape[3],\
                                       saturation_dataset_test.shape[4]])
    
saturation_predict_error = np.zeros([saturation_dataset_test.shape[0],saturation_dataset_test.shape[1],\
                                 saturation_dataset_test.shape[2],saturation_dataset_test.shape[3],\
                                 saturation_dataset_test.shape[4]])
    
saturation_predict_error_case_min = np.zeros(saturation_dataset_test.shape[0])
saturation_predict_error_case_max = np.zeros(saturation_dataset_test.shape[0])
saturation_predict_error_case_mean = np.zeros(saturation_dataset_test.shape[0])
saturation_predict_error_month_min = np.zeros(saturation_dataset_test.shape[4])
saturation_predict_error_month_max = np.zeros(saturation_dataset_test.shape[4])
saturation_predict_error_month_mean = np.zeros(saturation_dataset_test.shape[4])

plot_month = 50
model_ysizeHalf = 16
plot_layer = model_ysizeHalf
plot_cases = test_cases

from satFuncFittingSingleTime import satFuncFittingSingleTime
rePerm_drainage_test,capi_drainage_test = satFuncFittingSingleTime(saturation_predict_recovered[:,:,:,:,0],drainage_satTable_test)

plt.rcParams['figure.dpi'] = 144
for month in range(1,50):
    X_test[:,:,:,:,0] = perm_xyz_test[:,:,:,:,0]*\
    rePerm_drainage_test
    X_test[:,:,:,:,1] = perm_xyz_test[:,:,:,:,1]*\
    rePerm_drainage_test
    X_test[:,:,:,:,2] = perm_xyz_test[:,:,:,:,2]*\
    rePerm_drainage_test  
    
    X_test[:,:,:,:,4] = inject_gir_month_test[:,:,:,:,month-1]
    
    X_test[:,:,:,:,5] = time_test[:,:,:,:,month]  
           
    X_test[:,:,:,:,9] = capi_drainage_test
    
    X_test_roll = np.rollaxis(X_test,4,1)        
    y_test = saturation_dataset_test[:,:,:,:,month]
        
    targets_test = Variable(torch.FloatTensor(y_test.shape[1], y_test.shape[2], y_test.shape[3]))
    inputs_test = Variable(torch.FloatTensor(X_test_roll.shape[1], X_test_roll.shape[2], \
                                             X_test_roll.shape[3], X_test_roll.shape[4]))
    # netG.eval()
    # netG_model.eval()    
    for i, testdata in enumerate(X_test_roll, 0):
        #inputs_cpu, targets_cpu = test
        testdata = torch.tensor(testdata)
        targetdata = torch.tensor(y_test[i,:,:,:])
        inputs_test.data.copy_(testdata.float())
        targets_test.data.copy_(targetdata.float())
        
        outputs = netG_model(inputs_test.unsqueeze(0))
        
        outputs = torch.squeeze(outputs,1)
        outputs = torch.squeeze(outputs,0)
        
        outputs_cpu = outputs.data.cpu().numpy()
        
        X_test_case = testdata.detach().numpy()
        y_test_case = targetdata.detach().numpy()
        y_predict_case = outputs.detach().numpy()
        
        ## I used relative permeability 0.86 as the maximum for normalization
        x_permeability_recovered = (0.86)*X_test_case[0,:,:,:]*(perm_xyz_max-perm_xyz_min) + perm_xyz_min
        
        y_permeability_recovered = (0.86)*X_test_case[1,:,:,:]*(perm_xyz_max-perm_xyz_min) + perm_xyz_min
        
        z_permeability_recovered = (0.86)*X_test_case[2,:,:,:]*(perm_xyz_max-perm_xyz_min) + perm_xyz_min
        
        porosity_recovered = X_test_case[3,:,:,:]*(porosity_max-porosity_min) + porosity_min

        y_test_case_recovered = y_test_case 
        y_predict_case_recovered = y_predict_case

        saturation_predict_recovered[i,:,:,:,month] = y_predict_case_recovered
        saturation_test_recovered[i,:,:,:,month] = y_test_case_recovered
               
        x_permeability_recovered_array[i,:,:,:] = x_permeability_recovered
        y_permeability_recovered_array[i,:,:,:] = y_permeability_recovered
        z_permeability_recovered_array[i,:,:,:] = z_permeability_recovered
        porosity_recovered_array[i,:,:,:] = porosity_recovered
        
        saturation_predict_error[i,:,:,:,month] = abs(y_predict_case_recovered - y_test_case_recovered)

        if i < plot_cases:  
            fig, axs = plt.subplots(2, 2, figsize=(12, 8),constrained_layout=True)
            axs[0,0].spines['bottom'].set_linewidth(2)
            axs[0,0].spines['left'].set_linewidth(2)
            axs[0,0].spines['right'].set_linewidth(2)
            axs[0,0].spines['top'].set_linewidth(2) 
            
            axs[0,1].spines['bottom'].set_linewidth(2)
            axs[0,1].spines['left'].set_linewidth(2)
            axs[0,1].spines['right'].set_linewidth(2)
            axs[0,1].spines['top'].set_linewidth(2) 
            
            axs[1,0].spines['bottom'].set_linewidth(2)
            axs[1,0].spines['left'].set_linewidth(2)
            axs[1,0].spines['right'].set_linewidth(2)
            axs[1,0].spines['top'].set_linewidth(2) 
            
            axs[1,1].spines['bottom'].set_linewidth(2)
            axs[1,1].spines['left'].set_linewidth(2)
            axs[1,1].spines['right'].set_linewidth(2)
            axs[1,1].spines['top'].set_linewidth(2) 
            ####################################################################################################
            h1=axs[0,0].pcolormesh(x_coordinates[:,plot_layer,:], z_coordinates[:,plot_layer,:],  x_permeability_recovered[:,plot_layer,:],\
                                   cmap='rainbow')                
            axs[0,0].set_title('X effective permeability (mD)',fontsize = 20)
            #axs[0,0].axis('off')
            axs[0,0].xaxis.set_ticks([341000, 343000, 344500])
            axs[0,0].set_xticklabels([341000, 343000, 344500], fontsize = 20)
            axs[0,0].yaxis.set_ticks([-6500, -6300, -6100])
            axs[0,0].set_yticklabels([-6500, -6300, -6100], fontsize = 20)
            axs[0,0].set_xlabel('X (ft)',fontsize = 20)
            axs[0,0].set_ylabel('Depth (ft)',fontsize = 20)
            cb1=plt.colorbar(h1)
            # cb1.ax.yaxis.set_major_locator(MultipleLocator(100))
            # cb1.ax.yaxis.set_minor_locator(MultipleLocator(100))
            cb1.ax.tick_params(labelsize=18)
            
            h2=axs[0,1].pcolormesh(x_coordinates[:,plot_layer,:], z_coordinates[:,plot_layer,:], y_test_case_recovered[:,plot_layer,:], cmap='seismic', vmin=0, vmax=1)
            axs[0,1].set_title('Simulation',fontsize = 20)
            axs[0,1].xaxis.set_ticks([341000, 343000, 344500])
            axs[0,1].set_xticklabels([341000, 343000, 344500], fontsize = 20)
            axs[0,1].yaxis.set_ticks([-6500, -6300, -6100])
            axs[0,1].set_yticklabels([-6500, -6300, -6100], fontsize = 20)
            axs[0,1].set_xlabel('X (ft)',fontsize = 20)
            axs[0,1].set_ylabel('Depth (ft)',fontsize = 20)
            cb2=plt.colorbar(h2)
            # cb2.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            # cb2.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
            cb2.ax.tick_params(labelsize=18)
            
            h3=axs[1,0].pcolormesh(x_coordinates[:,plot_layer,:], z_coordinates[:,plot_layer,:], y_predict_case_recovered[:,plot_layer,:], cmap='seismic', vmin=0, vmax=1)
            axs[1,0].set_title('Prediction',fontsize = 20)
            axs[1,0].xaxis.set_ticks([341000, 343000, 344500])
            axs[1,0].set_xticklabels([341000, 343000, 344500], fontsize = 20)
            axs[1,0].yaxis.set_ticks([-6500, -6300, -6100])
            axs[1,0].set_yticklabels([-6500, -6300, -6100], fontsize = 20)
            axs[1,0].set_xlabel('X (ft)',fontsize = 20)
            axs[1,0].set_ylabel('Depth (ft)',fontsize = 20)
            cb3=plt.colorbar(h3)
            # cb3.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            # cb3.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
            cb3.ax.tick_params(labelsize=18)
            
            h4=axs[1,1].pcolormesh(x_coordinates[:,plot_layer,:], z_coordinates[:,plot_layer,:], abs(y_predict_case_recovered[:,plot_layer,:] - y_test_case_recovered[:,plot_layer,:]), cmap='seismic', vmin=0, vmax=1)
            axs[1,1].set_title('Difference',fontsize = 20)
            axs[1,1].xaxis.set_ticks([341000, 343000, 344500])
            axs[1,1].set_xticklabels([341000, 343000, 344500], fontsize = 20)
            axs[1,1].yaxis.set_ticks([-6500, -6300, -6100])
            axs[1,1].set_yticklabels([-6500, -6300, -6100], fontsize = 20)
            axs[1,1].set_xlabel('X (ft)',fontsize = 20)
            axs[1,1].set_ylabel('Depth (ft)',fontsize = 20)
            cb4=plt.colorbar(h4)
            # cb4.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            # cb4.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
            cb4.ax.tick_params(labelsize=18)
            
            # plt.suptitle(('month '+str(month+1)+' - newCase #'+str((i+1))+', CO$_2$ saturation @ CCS1, Dec 2015'),fontsize = 24)     
            plt.suptitle(('Month '+str(month+1)+', CO$_2$ saturation @ CCS1'),fontsize = 24)  
            plt.savefig('./results_test_saturation/newCase_' \
                       + str((i+1)) +'_month_'+str(month+1)+ '_xzView.png')  
            plt.show()
            
    rePerm_drainage_test,capi_drainage_test = satFuncFittingSingleTime(saturation_predict_recovered[:,:,:,:,month],drainage_satTable_test)
        

popt = []
popt.append(1)
popt.append(0)
    
plt.rcParams['ytick.labelsize'] = 38
plt.rcParams['xtick.labelsize'] = 38
plt.rcParams['figure.dpi'] = 144  

line_point = np.arange(0,0.674,0.01)
for i in range(test_cases):
    fig, axs = plt.subplots(2, 3, figsize=(29,18),constrained_layout=True) 
    axs[0,0].spines['bottom'].set_linewidth(2)
    axs[0,0].spines['left'].set_linewidth(2)
    axs[0,0].spines['right'].set_linewidth(2)
    axs[0,0].spines['top'].set_linewidth(2) 
    
    axs[0,1].spines['bottom'].set_linewidth(2)
    axs[0,1].spines['left'].set_linewidth(2)
    axs[0,1].spines['right'].set_linewidth(2)
    axs[0,1].spines['top'].set_linewidth(2) 
    
    axs[0,2].spines['bottom'].set_linewidth(2)
    axs[0,2].spines['left'].set_linewidth(2)
    axs[0,2].spines['right'].set_linewidth(2)
    axs[0,2].spines['top'].set_linewidth(2) 
    
    axs[1,0].spines['bottom'].set_linewidth(2)
    axs[1,0].spines['left'].set_linewidth(2)
    axs[1,0].spines['right'].set_linewidth(2)
    axs[1,0].spines['top'].set_linewidth(2) 
    
    axs[1,1].spines['bottom'].set_linewidth(2)
    axs[1,1].spines['left'].set_linewidth(2)
    axs[1,1].spines['right'].set_linewidth(2)
    axs[1,1].spines['top'].set_linewidth(2) 
    
    axs[1,2].spines['bottom'].set_linewidth(2)
    axs[1,2].spines['left'].set_linewidth(2)
    axs[1,2].spines['right'].set_linewidth(2)
    axs[1,2].spines['top'].set_linewidth(2) 
    
    h1=axs[0,0].scatter(saturation_test_recovered[i,:,:,:,4],\
        saturation_predict_recovered[i,:,:,:,4], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,4].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,4].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[0,0].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[0,0].set_title('Month 5',fontsize = 38)
    axs[0,0].set_xlim(0, 0.69)
    # axs[0,0].set_ylim(0, 0.82)
    axs[0,0].set_ylim(0, 1.2)
    axs[0,0].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[0,0].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[0,0].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,0].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,0].set_xlabel('True',fontsize = 38)
    axs[0,0].set_ylabel('Prediction',fontsize = 38)      
    axs[0,0].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  
  
    h2=axs[0,1].scatter(saturation_test_recovered[i,:,:,:,9],\
        saturation_predict_recovered[i,:,:,:,9], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,9].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,9].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[0,1].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[0,1].set_title('Month 10',fontsize = 38)
    axs[0,1].set_xlim(0, 0.69)
    # axs[0,1].set_ylim(0, 0.82)
    axs[0,1].set_ylim(0, 1.2)
    axs[0,1].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[0,1].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[0,1].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,1].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,1].set_xlabel('True',fontsize = 38)
    axs[0,1].set_ylabel('Prediction',fontsize = 38)
    axs[0,1].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  
  
    h3=axs[0,2].scatter(saturation_test_recovered[i,:,:,:,19],\
        saturation_predict_recovered[i,:,:,:,19], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,19].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,19].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[0,2].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[0,2].set_title('Month 20',fontsize = 38)
    axs[0,2].set_xlim(0, 0.69)
    axs[0,2].set_ylim(0, 0.82)
    axs[0,2].set_ylim(0, 1.2)
    axs[0,2].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[0,2].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[0,2].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,2].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0,2].set_xlabel('True',fontsize = 38)
    axs[0,2].set_ylabel('Prediction',fontsize = 38)
    axs[0,2].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  

    h4=axs[1,0].scatter(saturation_test_recovered[i,:,:,:,29],\
        saturation_predict_recovered[i,:,:,:,29], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,29].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,29].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[1,0].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[1,0].set_title('Month 30',fontsize = 38)
    axs[1,0].set_xlim(0, 0.69)
    axs[1,0].set_ylim(0, 0.82)
    axs[1,0].set_ylim(0, 1.2)
    axs[1,0].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[1,0].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[1,0].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,0].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,0].set_xlabel('True',fontsize = 38)
    axs[1,0].set_ylabel('Prediction',fontsize = 38)
    axs[1,0].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  
 
    h5=axs[1,1].scatter(saturation_test_recovered[i,:,:,:,39],\
        saturation_predict_recovered[i,:,:,:,39], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,39].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,39].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[1,1].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[1,1].set_title('Month 40',fontsize = 38)
    axs[1,1].set_xlim(0, 0.69)
    axs[1,1].set_ylim(0, 0.82)
    axs[1,1].set_ylim(0, 1.2)
    axs[1,1].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[1,1].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[1,1].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,1].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,1].set_xlabel('True',fontsize = 38)
    axs[1,1].set_ylabel('Prediction',fontsize = 38)
    axs[1,1].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  
 
    h6=axs[1,2].scatter(saturation_test_recovered[i,:,:,:,49],\
        saturation_predict_recovered[i,:,:,:,49], c='darkorange', s=120)
        
    saturation_test_temp = saturation_test_recovered[i,:,:,:,49].flatten()
    saturation_predict_temp = saturation_predict_recovered[i,:,:,:,49].flatten()    
    res_ydata  = saturation_predict_temp - saturation_test_temp
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((saturation_predict_temp - np.mean(saturation_predict_temp))**2)
    r_squared  = 1 - (ss_res / ss_tot)
        
    axs[1,2].plot(line_point, line_point, linewidth = 7, color = 'k', label = f'$R^2$ = {r_squared:.3f}')
    axs[1,2].set_title('Month 50',fontsize = 38)
    axs[1,2].set_xlim(0, 0.69)
    # axs[1,2].set_ylim(0, 0.82)
    axs[1,2].set_ylim(0, 1.2)
    axs[1,2].xaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
    axs[1,2].set_xticklabels([0.0, 0.2, 0.4, 0.6])
    axs[1,2].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,2].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1,2].set_xlabel('True',fontsize = 38)
    axs[1,2].set_ylabel('Prediction',fontsize = 38)
    axs[1,2].legend(loc = 2, fontsize = 38, handletextpad=0.2, frameon=False)  
          
    # plt.suptitle(('Test case '+str(i+1)+', CO$_2$ saturation @ CCS1'),fontsize = 24) 
    plt.suptitle(('CO$_2$ saturation @ CCS1'),fontsize = 42)  
    plt.savefig('./results_test_saturation/newCase_' \
            + str((i+1)) + '_comparison.png')  
    plt.show()

saturation_predict_recovered_min = saturation_predict_recovered.min(axis=3)
saturation_predict_recovered_min = saturation_predict_recovered_min.min(axis=2)
saturation_predict_recovered_min = saturation_predict_recovered_min.min(axis=1)

np.save("./error_recordFiles/saturation_predict_case_min.npy", saturation_predict_recovered_min)

saturation_predict_recovered_max = saturation_predict_recovered.max(axis=3)
saturation_predict_recovered_max = saturation_predict_recovered_max.max(axis=2)
saturation_predict_recovered_max = saturation_predict_recovered_max.max(axis=1)

np.save("./error_recordFiles/saturation_predict_case_max.npy", saturation_predict_recovered_max)

saturation_test_recovered_min = saturation_test_recovered.min(axis=3)
saturation_test_recovered_min = saturation_test_recovered_min.min(axis=2)
saturation_test_recovered_min = saturation_test_recovered_min.min(axis=1)

saturation_test_recovered_max = saturation_test_recovered.max(axis=3)
saturation_test_recovered_max = saturation_test_recovered_max.max(axis=2)
saturation_test_recovered_max = saturation_test_recovered_max.max(axis=1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['figure.dpi'] = 144  

fig, axs = plt.subplots(2, 2, figsize=(16, 10),constrained_layout=True)
axs[0,0].spines['bottom'].set_linewidth(2)
axs[0,0].spines['left'].set_linewidth(2)
axs[0,0].spines['right'].set_linewidth(2)
axs[0,0].spines['top'].set_linewidth(2) 
    
axs[0,1].spines['bottom'].set_linewidth(2)
axs[0,1].spines['left'].set_linewidth(2)
axs[0,1].spines['right'].set_linewidth(2)
axs[0,1].spines['top'].set_linewidth(2) 
    
axs[1,0].spines['bottom'].set_linewidth(2)
axs[1,0].spines['left'].set_linewidth(2)
axs[1,0].spines['right'].set_linewidth(2)
axs[1,0].spines['top'].set_linewidth(2) 
    
axs[1,1].spines['bottom'].set_linewidth(2)
axs[1,1].spines['left'].set_linewidth(2)
axs[1,1].spines['right'].set_linewidth(2)
axs[1,1].spines['top'].set_linewidth(2) 
for i in range(2,test_cases):
    testCaseId = i - 1
    axs[0,0].plot(np.arange(2,51), saturation_test_recovered_max[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')
    axs[1,0].plot(np.arange(2,51), saturation_test_recovered_min[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')

axs[0,0].plot(np.arange(2,51), saturation_test_recovered_max[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[0,0].plot(np.arange(2,51), saturation_test_recovered_max[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 
    
axs[1,0].plot(np.arange(2,51), saturation_test_recovered_min[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[1,0].plot(np.arange(2,51), saturation_test_recovered_min[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 
        
axs[0,0].set_xlabel('Month',fontsize = 24)
axs[0,0].set_ylabel('Truth maximum',fontsize = 24)      
axs[0,0].legend(loc = 'lower right', ncol=4,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  

axs[1,0].set_xlabel('Month',fontsize = 24)
axs[1,0].set_ylabel('Truth minimum',fontsize = 24)      
axs[1,0].legend(loc = 'lower left', ncol=4,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  

for i in range(2,test_cases):
    testCaseId = i - 1
    axs[0,1].plot(np.arange(2,51), saturation_predict_recovered_max[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')
    axs[1,1].plot(np.arange(2,51), saturation_predict_recovered_min[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')

axs[0,1].plot(np.arange(2,51), saturation_predict_recovered_max[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[0,1].plot(np.arange(2,51), saturation_predict_recovered_max[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 
    
axs[1,1].plot(np.arange(2,51), saturation_predict_recovered_min[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[1,1].plot(np.arange(2,51), saturation_predict_recovered_min[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 

axs[0,1].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8])
axs[0,1].set_yticklabels([0.2, 0.4, 0.6, 0.8])        
axs[0,1].set_xlabel('Month',fontsize = 24)
axs[0,1].set_ylabel('Prediction maximum',fontsize = 24)      
axs[0,1].legend(loc = 'lower right', ncol=4,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  

# axs[1,1].set_ylim(-0.046, 0.02)
axs[1,1].yaxis.set_ticks([-0.14, -0.10, -0.06, -0.02])
axs[1,1].set_yticklabels([-0.14, -0.10, -0.06, -0.02])
axs[1,1].set_xlabel('Month',fontsize = 24)
axs[1,1].set_ylabel('Prediction minimum',fontsize = 24)      
axs[1,1].legend(loc = 'lower left', ncol=2,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  


# plt.suptitle(('CO$_2$ saturation maximum and minimum @ CCS1'),fontsize = 28)
plt.suptitle(('Saturation maximum and minimum of test cases'),fontsize = 28)  
plt.savefig('./results_test_saturation/minMaxSat_testCases.png')  
plt.show()


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['figure.dpi'] = 144  

fig, axs = plt.subplots(1, 2, figsize=(16, 6),constrained_layout=True)
axs[0].spines['bottom'].set_linewidth(2)
axs[0].spines['left'].set_linewidth(2)
axs[0].spines['right'].set_linewidth(2)
axs[0].spines['top'].set_linewidth(2) 
    
axs[1].spines['bottom'].set_linewidth(2)
axs[1].spines['left'].set_linewidth(2)
axs[1].spines['right'].set_linewidth(2)
axs[1].spines['top'].set_linewidth(2) 
for i in range(2,test_cases):
    testCaseId = i - 1
    axs[0].plot(np.arange(2,51), saturation_predict_recovered_max[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')
    axs[1].plot(np.arange(2,51), saturation_predict_recovered_min[i,1:],\
      linewidth = 3, color = colors[i-2], label = f'#{testCaseId}')

axs[0].plot(np.arange(2,51), saturation_predict_recovered_max[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[0].plot(np.arange(2,51), saturation_predict_recovered_max[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 
    
axs[1].plot(np.arange(2,51), saturation_predict_recovered_min[1,1:], \
  '--', linewidth = 3, color = '#bcbd22', label = '#9')    
axs[1].plot(np.arange(2,51), saturation_predict_recovered_min[0,1:], \
  ':', linewidth = 3, color = '#17becf', label = '#10') 
        
axs[0].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8])
axs[0].set_yticklabels([0.2, 0.4, 0.6, 0.8])    
axs[0].set_xlabel('Month',fontsize = 24)
axs[0].set_ylabel('Prediction maximum',fontsize = 24)      
axs[0].legend(loc = 'lower right', ncol=4,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  

# axs[1].set_ylim(-0.046, 0.02)
axs[1].yaxis.set_ticks([-0.14, -0.10, -0.06, -0.02])
axs[1].set_yticklabels([-0.14, -0.10, -0.06, -0.02])
axs[1].set_xlabel('Month',fontsize = 24)
axs[1].set_ylabel('Prediction minimum',fontsize = 24)      
axs[1].legend(loc = 'lower left', ncol=2,\
    handletextpad=0.2,labelspacing=0.2,columnspacing=0.2,
    fontsize = 20, frameon=False)  

# plt.suptitle(('CO$_2$ saturation maximum and minimum @ CCS1'),fontsize = 28)
plt.suptitle(('Saturation maximum and minimum of test cases'),fontsize = 28)  
plt.savefig('./results_test_saturation/minMaxSat_prediction.png')  
plt.show()
