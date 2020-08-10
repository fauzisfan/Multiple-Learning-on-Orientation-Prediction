"""
Created on Fri Feb 28 16:52:45 2020

@author: Isfan
"""

import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import norm

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
from scipy import signal
# from disolve import remdiv
# modularized library import
import sys
sys.path.append('/gdrive/My Drive/Colab Notebooks/Motion prediction')

from preparets import preparets
from train_test_split import train_test_split_tdnn
#from calc_future_orientation import calc_future_orientation
from eul2quat_bio import eul2quat_bio
from quat2eul_bio import quat2eul_bio
#from ann_prediction import ann_prediction
from cap_prediction import cap_prediction
from crp_prediction import crp_prediction
from nop_prediction import nop_prediction
from solve_discontinuity import solve_discontinuity
from rms import rms    
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat   
import os
import math
from sklearn.externals import joblib

@tf.function
def model_func(model, x_train, t_train):
    y_pred = model(x_train)
    loss = tf.losses.MeanSquaredError()(y_pred,t_train)
    return y_pred, loss

def copy_model(model, x):
    '''
        To copy weight/variable model to the next model
    '''
    copied_model = tf.keras.models.clone_model(model)
    copied_model.forward = model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# In[5]:

anticipation_time = 300
TEST_SIZE = 0.7
TRAIN_SIZE = 1 - TEST_SIZE
'''
#####   SYSTEM INITIALIZATION    #####
'''
def dataCaller(filename):
    tf.compat.v1.reset_default_graph()
    
    tf.compat.v1.set_random_seed(2)
    
    np.random.seed(2)
    
    
    #parser = argparse.ArgumentParser(description='Offline Motion Prediction')
    #parser.add_argument('-a', '--anticipation', default=300, type=int)
    
    #args = parser.parse_args()
    
    try:
        stored_df = pd.read_csv(filename)
        train_gyro_data = np.array(stored_df[['angular_vec_x', 'angular_vec_y', 'angular_vec_z']], dtype=np.float)
        train_acce_data = np.array(stored_df[['acceleration_x', 'acceleration_y', 'acceleration_z']], dtype=np.float)
        train_magn_data = np.array(stored_df[['magnetic_x', 'magnetic_y', 'magnetic_z']], dtype=np.float)
        train_eule_data = np.array(stored_df[['input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw']], dtype=np.float)
        train_time_data = np.array(stored_df['timestamp'], dtype=np.float)
        train_time_data = train_time_data / 705600000
        train_data_id = stored_df.shape[0]
        print('\nSaved data loaded...\n')
        
    except:
        raise
    
    # In[6]:
    
    
    '''
    #####   데이터 로드    #####
    '''
    print('Training data preprocessing is started...')
    
    
    # Remove zero data from collected training data
    system_rate = round((train_data_id+1)/float(np.max(train_time_data) - train_time_data[0]))
    idle_period = int(2 * system_rate)
    train_gyro_data = train_gyro_data* 180/ np.pi
    train_eule_data = train_eule_data * 180 / np.pi
    train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
    train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float), train_alfa_data])
    train_alfa_data = train_alfa_data * np.pi / 180
    
    
    """Velocity data"""
    train_velocity_data = np.diff(train_eule_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
    train_velocity_data = np.row_stack([np.zeros(shape=(1, train_velocity_data.shape[1]), dtype=np.float), train_velocity_data])
    
    """Acceleration diff"""
    train_acce_diff_data = np.diff(train_velocity_data, axis =0)/ np.diff(train_time_data, axis=0).reshape(-1, 1)
    train_acce_diff_data =np.row_stack([np.zeros(shape=(1, train_acce_diff_data.shape[1]), dtype=np.float), train_acce_diff_data])
    # Calculate the head orientation
    #train_gyro_data = train_gyro_data * np.pi / 180
    train_acce_data = train_acce_data / 9.8
    train_magn_data = train_magn_data
    
    train_eule_data = solve_discontinuity(train_eule_data)
    
    train_quat_data = eul2quat_bio(train_eule_data)
    train_quat2eul_data = quat2eul_bio(train_quat_data)
    
    # Create data frame of all features and smoothing
    sliding_window_time = 100
    sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))
    
    ann_feature = np.column_stack([train_eule_data, 
                                   train_gyro_data, 
                                   train_alfa_data, 
    #                               train_magn_data,
                                   ])
        
    feature_name = ['pitch', 'roll', 'yaw', 
                    'gX', 'gY', 'gZ', 
                    'aX', 'aY', 'aZ', 
    #                'mX', 'mY', 'mZ', 
    #                'EMG1', 'EMG2', 'EMG3', 'EMG4',
                    ]
    
    ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
    #ann_feature_df = ann_feature_df.rolling(sliding_window_size, min_periods=1).mean()
    
    
    # Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
    #anticipation_time = args.anticipation  # 앞 셀에서 정의함
    anticipation_size = int(np.round(anticipation_time * system_rate / 1000))
    print('anticipation size = ', anticipation_size)
    
    #lhood1 = 100
    #lhood2 = 200
    #lhood3 = 300
    #lhood1_size = int(np.round(lhood1 * system_rate / 1000))
    #lhood2_size = int(np.round(lhood2 * system_rate / 1000))
    #lhood3_size = int(np.round(lhood3 * system_rate / 1000))
    
    spv_name = ['pitch', 'roll', 'yaw']
    target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)
    input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)
    
    input_nm = len(input_series_df.columns)
    target_nm = len(target_series_df.columns)
    
    
    # In[17]:
    
    
    '''
    #####   NN 입력 데이터 준비    #####
    '''
    
    # Neural network parameters
    DELAY_SIZE = int(100 * (system_rate / 1000))  # 어떤 용도? 샘플 윈도우?
    
    print("DELAY_SIZE =", DELAY_SIZE)
    
    # Variables
    # TRAINED_MODEL_NAME = './best4_net3'
    
    # Import datasets
    input_series = np.array(input_series_df)
    target_series = np.array(target_series_df)
    
    
    """"""""" New Preprocessong """""""""
    ## Split training and testing data
    #x_seq, t_seq = preparets(input_series, target_series, DELAY_SIZE)
    #data_length = x_seq.shape[0]
    #scaler = preprocessing.StandardScaler().fit(training_series)	# fit saves normalization coefficient into scaler
    #
    #x_seq, t_seq = remdiv(x_seq, t_seq, DELAY_SIZE)
    #
    ##Normalize training data, then save the normalization coefficient
    #for i in range(0,len(x_seq)):
    #    x_seq[i,:,:] = scaler.transform(x_seq[i,:,:])
    #    
    #x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, target_series, TEST_SIZE)
    
    """"""""" Old Preprocessong """""""""
    # Save it
    # scaler_file = "my_scaler3.save"
    normalizer = preprocessing.StandardScaler()
    #Get normalized based data on the 
    tempNorm = normalizer.fit(input_series)
    # joblib.dump(tempNorm, scaler_file)
    
    # # Load it 
    # tempNorm = joblib.load(scaler_file) 
    #Normalizer used on input series
    input_norm = tempNorm.transform(input_series)
    
    
    # Reformat the input into TDNN format
    x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
    data_length = x_seq.shape[0]
    print('Anticipation time: {}ms\n'.format(anticipation_time))
    
    
    # Reset the whole tensorflow graph
    tf.compat.v1.reset_default_graph()
    
    
    # Split training and testing data
    x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, TEST_SIZE)
    
    x_train = tf.convert_to_tensor(x_train)
    t_train = tf.convert_to_tensor(t_train)
    x_test = tf.convert_to_tensor(x_test)
    t_test = tf.convert_to_tensor(t_test)
    x_seq = tf.convert_to_tensor(x_seq)
    # t_seq = tf.convert_to_tensor(t_seq)
    
    
    return x_seq, t_seq, x_train, t_train, x_test, t_test, train_velocity_data, train_time_data, train_eule_data, train_acce_data,train_gyro_data,train_alfa_data, input_nm, target_nm, anticipation_size, DELAY_SIZE


tf.compat.v1.set_random_seed(2)

np.random.seed(2)

# In[20]:
x_seq, t_seq, x_train, t_train, x_test, t_test, train_velocity_data, train_time_data, train_eule_data, train_acce_data,train_gyro_data,train_alfa_data, input_nm, target_nm, anticipation_size, DELAY_SIZE = dataCaller('20200420_scene(3)_user(1).csv')
# x_seq2, t_seq2, x_train2, t_train2, x_test2, t_test2, _, _, _, _, _, _, _, _, _, _= dataCaller('20200220_scene(3)_user(2).csv')

data_length = x_seq.shape[0]

vel_seq = train_velocity_data[DELAY_SIZE:-anticipation_size]
vel_test = vel_seq[int((1-TEST_SIZE)*data_length):]
vel_train = vel_seq[:int((1-TEST_SIZE)*data_length)]

midVel = np.nanpercentile(np.abs(train_velocity_data),50, axis = 0)
avgVel = np.nanmean(np.abs(train_velocity_data), axis=0)

tempNOP = train_eule_data[DELAY_SIZE:-anticipation_size]
tempNOP_test = tempNOP[int((1-TEST_SIZE)*data_length):]

initializer = tf.keras.initializers.lecun_uniform(seed=None)
# Define TDNN model
model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(DELAY_SIZE, input_nm)),
            tf.keras.layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), kernel_initializer='lecun_uniform', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(9, activation=tf.nn.relu, kernel_initializer = initializer, 
                                  kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            ])

start_time = time.time()

'''Define Learning parameter '''
n_epochs=1000
learning_rate = 0.01
optimizer_pretrained = tf.keras.optimizers.Adam(learning_rate = learning_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.02)
pretrained_batch = 1 
batch_length = 150
online_length = len(t_test)
threshold = 0.3
vel_threshold = 1.55
num_batch = []
MAE_batch = np.zeros((0,1), dtype= float)

method = "OnL"
scheme = "soft"

if(method == "OnL"):
    print("Method : " + method)
    '''
    #####   ONLINE LEARNING    #####
    '''
    ## For Pre-trained model
    model_copy = copy_model(model,x_train)
    
    x_train_batch = x_train[0:int(len(x_train)*pretrained_batch)]
    t_train_batch = t_train[0:int(len(t_train)*pretrained_batch)]
    for epoch in range(n_epochs):
        with tf.GradientTape() as update:
            _, loss = model_func(model_copy, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
        gradients = update.gradient(loss, model_copy.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
        optimizer_pretrained.apply_gradients(zip(gradients, model_copy.trainable_variables))#apply optimizer to compute gradients and update weight
        
    y_pred, test_mse  = model_func(model_copy, x_test[0:batch_length], t_test[0:batch_length])
    test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[0:batch_length])
    print("(loss) pre-trained Model ")
    print(test_mae)
        
    ## For Online Learning
    # model_online = copy_model(model_copy,x_train_batch)
    for j in range(int(online_length/batch_length-1)):
        
        x_train_batch = x_test[0:(j+1)*(batch_length)]
        t_train_batch = t_test[0:(j+1)*(batch_length)]
        
        model_online = copy_model(model_copy,x_train_batch)
        for epoch in range(100):
            with tf.GradientTape() as update:
                _, loss = model_func(model_online, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
            gradients = update.gradient(loss, model_online.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
            optimizer.apply_gradients(zip(gradients, model_online.trainable_variables))#apply optimizer to compute gradients and update weight
        
        y_pred, test_mse  = model_func(model_online, x_test[(j+1)*(batch_length):(j+2)*(batch_length)], t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        y_pred = np.array(y_pred)
        
        '''Online Switching'''
        m = 0
        if (scheme == "hard"):
        #   Hard Switching
            # vel_threshold =  np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),15, axis = 0)
        
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
            #   Switch to NOP if velocity under 8deg/s
                for l in range(0,3):
                    if (np.abs(velocity_onedata[l])<vel_threshold):
                        y_pred[m,l] = tempNOP_test[k,l]
                m = m+1
        elif(scheme == "soft"):
        #   Soft Switching
            # midVel = np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),50, axis = 0)
            # avgVel = np.nanmean(np.abs(vel_test[0:(j+1)*(batch_length)]), axis=0)
            
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
                for l in range(0,3):
                    xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
                    alfa = sigmoid(xin)
                    y_pred[m,l] = alfa*y_pred[m,l] + (1-alfa)*tempNOP_test[k,l]
                m = m+1
                
        test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        
        # x_coba = tf.concat([x_train,x_train_batch],0)
        # t_coba = tf.concat([t_train,t_train_batch],0)
        # y_pred, test_mse  = model_func(model_online, x_coba, t_coba)
        # test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_coba)
        
        num_batch.append(j)
        MAE_batch = np.concatenate((MAE_batch, np.reshape(test_mae,(1,1))), axis =0)
        print("(loss) batch ",j)
        print(test_mae)
        if (test_mae<threshold) or j==(int(online_length/batch_length)-1):
            print("Training Done. Final MAE :")
            print(test_mae)
            break
            
    
    # Smoothing with Savgol filtering
    #y_out = signal.savgol_filter(y_out, SG_window_size, 3, axis=0)
    
    y_out = model_online(x_seq)
    y_out = np.array(y_out)
    
elif(method == "OffL"):
    print("Method : " + method)
    '''
    #####   OFFLINE LEARNING    #####
    '''
    model_offline = copy_model(model,x_train)
    for epoch in range(n_epochs):
        with tf.GradientTape() as update:
            _, loss = model_func(model_offline, x_train,t_train)#forward pass from input > multiply by variable > output & loss
        gradients = update.gradient(loss, model_offline.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
        optimizer.apply_gradients(zip(gradients, model_offline.trainable_variables))#apply optimizer to compute gradients and update weight
    
    y_out = model_offline(x_seq)
    y_out = np.array(y_out)

elif(method == "TOE"):
    print("Method : " + method)
    '''
    #####   TOE    #####
    '''
    
    ## For Pre-trained model
    model_copy = copy_model(model,x_train)
    
    x_train_batch = x_train
    t_train_batch = t_train
    for epoch in range(n_epochs):
        with tf.GradientTape() as update:
            _, loss = model_func(model_copy, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
        gradients = update.gradient(loss, model_copy.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
        optimizer_pretrained.apply_gradients(zip(gradients, model_copy.trainable_variables))#apply optimizer to compute gradients and update weight
        
    y_pred, test_mse  = model_func(model_copy, x_test[0:batch_length], t_test[0:batch_length])
    test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[0:batch_length])
    print("(loss) pre-trained Model ")
    print(test_mae)
        
    ## For Online Learning
    # model_online = copy_model(model_copy,x_train_batch)
    for j in range(int(online_length/batch_length-1)):
        
        x_train_batch = x_test[0:(j+1)*(batch_length)]
        t_train_batch = t_test[0:(j+1)*(batch_length)]
        
        x_toe = tf.concat([x_train,x_train_batch],0)
        t_toe = tf.concat([t_train,t_train_batch],0)
        
        model_toe = copy_model(model_copy,x_toe)
        for epoch in range(100):
            with tf.GradientTape() as update:
                _, loss = model_func(model_toe, x_toe,t_toe)#forward pass from input > multiply by variable > output & loss
            gradients = update.gradient(loss, model_toe.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
            optimizer.apply_gradients(zip(gradients, model_toe.trainable_variables))#apply optimizer to compute gradients and update weight
        
        y_pred, test_mse  = model_func(model_toe, x_test[(j+1)*(batch_length):(j+2)*(batch_length)], t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        y_pred = np.array(y_pred)
        
        '''Online Switching'''
        m = 0
        if (scheme == "hard"):
        #   Hard Switching
            # vel_threshold =  np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),15, axis = 0)
            
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
            #   Switch to NOP if velocity under 8deg/s
                for l in range(0,3):
                    if (np.abs(velocity_onedata[l])<vel_threshold):
                        y_pred[m,l] = tempNOP_test[k,l]
                m = m+1
        elif (scheme == "soft"):
        #   Soft Switching
            # midVel = np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),50, axis = 0)
            # avgVel = np.nanmean(np.abs(vel_test[0:(j+1)*(batch_length)]), axis=0)
        
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
                for l in range(0,3):
                    xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
                    alfa = sigmoid(xin)
                    y_pred[m,l] = alfa*y_pred[m,l] + (1-alfa)*tempNOP_test[k,l]
                m = m+1
    
        test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
    
        # x_coba = tf.concat([x_train,x_train_batch],0)
        # t_coba = tf.concat([t_train,t_train_batch],0)
        # y_pred, test_mse  = model_func(model_online, x_coba, t_coba)
        # test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_coba)
        
        num_batch.append(j)
        MAE_batch = np.concatenate((MAE_batch, np.reshape(test_mae,(1,1))), axis =0)
        print("(loss) batch ",j)
        print(test_mae)
        if (test_mae<threshold) or j==(int(online_length/batch_length)-1):
            print("Training Done. Final MAE :")
            print(test_mae)
            break
            
    
    # Smoothing with Savgol filtering
    #y_out = signal.savgol_filter(y_out, SG_window_size, 3, axis=0)
    
    y_out = model_toe(x_seq)
    y_out = np.array(y_out)

elif(method == "JT"):
    print("Method : " + method)
    '''
    #####   JOINT LEARNING    #####
    '''
    ## For Pre-trained model
    model_copy = copy_model(model,x_train)
    
    x_train_batch = x_train
    t_train_batch = t_train
    for epoch in range(n_epochs):
        with tf.GradientTape() as update:
            _, loss = model_func(model_copy, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
        gradients = update.gradient(loss, model_copy.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
        optimizer_pretrained.apply_gradients(zip(gradients, model_copy.trainable_variables))#apply optimizer to compute gradients and update weight
        
    y_pred, test_mse  = model_func(model_copy, x_test[0:batch_length], t_test[0:batch_length])
    test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[0:batch_length])
    print("(loss) pre-trained Model ")
    print(test_mae)
        
    ## For Online Learning
    model_joint = copy_model(model_copy,x_train_batch)
    for j in range(int(online_length/batch_length-1)):
        
        x_train_batch = x_test[j*(batch_length):(j+1)*(batch_length)]
        t_train_batch = t_test[j*(batch_length):(j+1)*(batch_length)]
        
        # model_joint = copy_model(model_copy,x_train_batch)
        for epoch in range(100):
            with tf.GradientTape() as update:
                _, loss = model_func(model_joint, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
            gradients = update.gradient(loss, model_joint.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
            optimizer.apply_gradients(zip(gradients, model_joint.trainable_variables))#apply optimizer to compute gradients and update weight
        
        y_pred, test_mse  = model_func(model_joint, x_test[(j+1)*(batch_length):(j+2)*(batch_length)], t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        y_pred = np.array(y_pred)
        
        '''Online Switching'''
        m = 0
        if (scheme == "hard"):
        #   Hard Switching
            # vel_threshold =  np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),15, axis = 0)
            
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
            #   Switch to NOP if velocity under 8deg/s
                for l in range(0,3):
                    if (np.abs(velocity_onedata[l])<vel_threshold):
                        y_pred[m,l] = tempNOP_test[k,l]
                m = m+1
        elif (scheme == "soft"):
        #   Soft Switching
            # midVel = np.nanpercentile(np.abs(vel_test[j*(batch_length):(j+1)*(batch_length)]),50, axis = 0)
            # avgVel = np.nanmean(np.abs(vel_test[j*(batch_length):(j+1)*(batch_length)]), axis=0)
            
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
                for l in range(0,3):
                    xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
                    alfa = sigmoid(xin)
                    y_pred[m,l] = alfa*y_pred[m,l] + (1-alfa)*tempNOP_test[k,l]
                m = m+1
            
        test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        
        # x_coba = tf.concat([x_train,x_train_batch],0)
        # t_coba = tf.concat([t_train,t_train_batch],0)
        # y_pred, test_mse  = model_func(model_online, x_coba, t_coba)
        # test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_coba)
        
        num_batch.append(j)
        MAE_batch = np.concatenate((MAE_batch, np.reshape(test_mae,(1,1))), axis =0)
        print("(loss) batch ",j)
        print(test_mae)
        if (test_mae<threshold) or j==(int(online_length/batch_length)-1):
            print("Training Done. Final MAE :")
            print(test_mae)
            break
            
    
    # Smoothing with Savgol filtering
    #y_out = signal.savgol_filter(y_out, SG_window_size, 3, axis=0)
    
    y_out = model_joint(x_seq)
    y_out = np.array(y_out)
    
elif(method == "TFS"):
    print("Method : " + method)
    '''
    #####   From Scratch Learning    #####
    '''
    ## For Online Learning
    # model_copy = copy_model(model,x_test)
    for j in range(int(online_length/batch_length-1)):
        
        x_train_batch = x_test[0:(j+1)*(batch_length)]
        t_train_batch = t_test[0:(j+1)*(batch_length)]
        
        model_copy = copy_model(model,x_train_batch)
        for epoch in range(100):
            with tf.GradientTape() as update:
                _, loss = model_func(model_copy, x_train_batch,t_train_batch)#forward pass from input > multiply by variable > output & loss
            gradients = update.gradient(loss, model_copy.trainable_variables) #backward pass : to trace computation graph to compute gradient for the weights. like minimize function : to find variable value
            optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))#apply optimizer to compute gradients and update weight
        
        y_pred, test_mse  = model_func(model_copy, x_test[(j+1)*(batch_length):(j+2)*(batch_length)], t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        y_pred = np.array(y_pred)
        
        '''Online Switching'''
        m = 0
        if (scheme == "hard"):
        #   Hard Switching
            # vel_threshold =  np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),15, axis = 0)
            
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
            #   Switch to NOP if velocity under 8deg/s
                for l in range(0,3):
                    if (np.abs(velocity_onedata[l])<vel_threshold):
                        y_pred[m,l] = tempNOP_test[k,l]
                m = m+1
        elif (scheme == "soft"):
        #   Soft Switching
            midVel = np.nanpercentile(np.abs(vel_test[0:(j+1)*(batch_length)]),50, axis = 0)
            avgVel = np.nanmean(np.abs(vel_test[0:(j+1)*(batch_length)]), axis=0)
        
            for k in range((j+1)*(batch_length), (j+2)*(batch_length)):
                velocity_onedata = vel_test[k]
                for l in range(0,3):
                    xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
                    alfa = sigmoid(xin)
                    y_pred[m,l] = alfa*y_pred[m,l] + (1-alfa)*tempNOP_test[k,l]
                m = m+1
    
        test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_test[(j+1)*(batch_length):(j+2)*(batch_length)])
        
        # x_coba = tf.concat([x_train,x_train_batch],0)
        # t_coba = tf.concat([t_train,t_train_batch],0)
        # y_pred, test_mse  = model_func(model_online, x_coba, t_coba)
        # test_mae = tf.losses.MeanAbsoluteError()(y_pred,t_coba)
        
        num_batch.append(j)
        MAE_batch = np.concatenate((MAE_batch, np.reshape(test_mae,(1,1))), axis =0)
        print("(loss) batch ",j)
        print(test_mae)
        if (test_mae<threshold) or j==(int(online_length/batch_length)-1):
            print("Training Done. Final MAE :")
            print(test_mae)
            break
            
    
    # Smoothing with Savgol filtering
    #y_out = signal.savgol_filter(y_out, SG_window_size, 3, axis=0)
    
    y_out = model_copy(x_seq)
    y_out = np.array(y_out)


'''Final test - Online Switching'''
m = 0
if (scheme=="hard"):
    #   Hard Switching
    for k in range(len(vel_seq)):
        velocity_onedata = vel_seq[k]
        if (method == "OffL"):
            if (k == 0):
                vel_threshold = velocity_onedata
            else:
                vel_threshold = np.nanpercentile(np.abs(vel_seq[:k]),15, axis = 0)
    #   Switch to NOP if velocity under 8deg/s
        for l in range(0,3):
            if (np.abs(velocity_onedata[l])<vel_threshold[l]):
                y_out[m,l] = tempNOP[k,l]
        m = m+1

elif (scheme=="soft"):
    # Soft Switching
    for k in range(len(vel_seq)):
        velocity_onedata = vel_seq[k]
        if (method == "OffL"):
            if (k == 0):
                vel_threshold = velocity_onedata
            else:
                midVel = np.nanpercentile(np.abs(vel_seq[:k]),50, axis = 0)
                avgVel = np.nanmean(np.abs(vel_seq[:k]), axis=0)
        for l in range(0,3):
            xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
            alfa = sigmoid(xin)
            y_out[m,l] = alfa*y_out[m,l] + (1-alfa)*tempNOP[k,l]
        m = m+1

'''
#####   ORIENTATION PREDICTION COMPARISON   #####
'''
# Recalibrate and align current time head orientation
euler_o = train_eule_data[DELAY_SIZE:-anticipation_size]
gyro_o = train_gyro_data[DELAY_SIZE:-anticipation_size]* np.pi / 180
alfa_o = train_alfa_data[DELAY_SIZE:-anticipation_size]
accel_o = train_acce_data[DELAY_SIZE:-anticipation_size]
# velocity_o = train_velocity_data[:-anticipation_size]

# Predict orientation
euler_pred_ann = y_out
euler_pred_cap = cap_prediction(euler_o, gyro_o, alfa_o, anticipation_time)
euler_pred_crp = crp_prediction(euler_o, gyro_o, anticipation_time)
euler_pred_nop = nop_prediction(euler_o, anticipation_time)

# Calculate prediction error
# Error is defined as difference between:
# Predicted head orientation
# Actual head orientation = Current head orientation shifted by s time
euler_ann_err = np.abs(euler_pred_ann[:-anticipation_size] - euler_o[anticipation_size:])
euler_cap_err = np.abs(euler_pred_cap[:-anticipation_size] - euler_o[anticipation_size:])
euler_crp_err = np.abs(euler_pred_crp[:-anticipation_size] - euler_o[anticipation_size:])
euler_nop_err = np.abs(euler_pred_nop[:-anticipation_size] - euler_o[anticipation_size:])
euler_ann_err2 = euler_pred_ann[:-anticipation_size] - euler_o[anticipation_size:]

#
euler_position_err_test = euler_ann_err[int((1-TEST_SIZE)*data_length):]
euler_cap_err_test = euler_cap_err[int((1-TEST_SIZE)*data_length):] 
euler_crp_err_test = euler_crp_err[int((1-TEST_SIZE)*data_length):] 
euler_nop_err_test = euler_nop_err[int((1-TEST_SIZE)*data_length):]


euler_ann_mae = np.nanmean(np.abs(euler_position_err_test), axis=0)
euler_cap_mae = np.nanmean(np.abs(euler_cap_err_test), axis=0)
euler_crp_mae = np.nanmean(np.abs(euler_crp_err_test), axis=0)
euler_nop_mae = np.nanmean(np.abs(euler_nop_err_test), axis=0)


#Calculate 99% Percentile
final_euler_99 = np.nanpercentile(euler_position_err_test,99, axis = 0)
final_euler_cap_99 = np.nanpercentile(euler_cap_err_test,99, axis = 0)
final_euler_crp_99 = np.nanpercentile(euler_crp_err_test,99, axis = 0)
final_euler_nop_99 = np.nanpercentile(euler_nop_err_test,99, axis = 0)

print('MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(euler_ann_mae[0], euler_ann_mae[1], euler_ann_mae[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(euler_nop_mae[0], euler_nop_mae[1], euler_nop_mae[2]))

print('\n99% MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_euler_99[0], final_euler_99[1], final_euler_99[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_euler_nop_99[0], final_euler_nop_99[1], final_euler_nop_99[2]))

''' Plot Something '''

plt.figure()
plt.plot(num_batch, MAE_batch[:,0], linewidth=1,color='magenta')
# plt.plot(num_batch, MAE_batch[:,1], linewidth=1,color='green')
# plt.plot(num_batch, MAE_batch[:,2],linewidth=1,color='blue')
# plt.legend(['X','Y','Z'])
plt.title('MAE vs Number of Batches')
plt.grid()
plt.xlabel('batch every ' + str(batch_length)+' data')
plt.ylabel('MAE Position (cm)')
plt.show(block=False)

plt.figure()
x = np.sort(MAE_batch[:,0])
y = np.arange(1, len(x)+1)/len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Percentage of MAE (degree)')
_ = plt.ylabel('likehood of Occurance')
plt.title('CDF of MAE test over batches')
plt.margins(0.02)
plt.show()


timestamp_plot = train_time_data[DELAY_SIZE:-2*anticipation_size]
time_offset = timestamp_plot[0]
timestamp_plot = np.array(timestamp_plot)-time_offset

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 2], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 2], linewidth=1, color='navy')
plt.legend(['ANN', 'NOP','Actual'])
plt.title('Orientation Prediction (Yaw)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 1], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 1], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 1], linewidth=1, color='navy')
plt.legend(['ANN', 'NOP','Actual'])
plt.title('Orientation Prediction (Roll)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 0], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 0], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 0], linewidth=1, color='navy')
plt.legend(['ANN', 'NOP','Actual'])
plt.title('Orientation Prediction (Pitch)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

"""Online Prediction"""
# import server

# aplikasi = server.App
# aplikasi()

# #Initialize temporary array

# # ann_pred_rt = y_sample[:,:3]
# # cap_pred_rt= np.zeros_like(euler_pred_cap)
# # crp_pred_rt = np.zeros_like(euler_pred_crp)
# # nop_pred_rt = np.zeros_like(euler_pred_nop)
# # ann_pred_rt = y_sample[:,:3]
# ann_pred_rt = np.zeros((0,3), dtype= float)
# ann_dummy_rt = np.zeros((0,3), dtype= float)
# cap_pred_rt = np.zeros((0,3), dtype= float)
# crp_pred_rt = np.zeros((0,3), dtype= float)
# nop_pred_rt = np.zeros((0,3), dtype= float)
# error_rt = np.zeros((0,3))

# k = 0
# m = 0

# import RealTimePlot as rtp

# graphShow = "movement"
# coordinate = "roll"

# x_seq_rt = np.zeros(shape=(DELAY_SIZE, input_nm))
# xd_seq_rt = np.zeros(shape=(DELAY_SIZE, input_nm))
# timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
# gyro_rt = np.zeros(shape=(2, 3), dtype= float)
# rt_counter = 0
# y_plot = np.zeros(5)
# midVel = np.nanpercentile(np.abs(velocity_o),50, axis = 0)
# avgVel = np.nanmean(np.abs(velocity_o), axis=0)

# with tf.Session() as new_sess:    
# 	model.load_weights(TRAINED_MODEL_NAME)
# 	for i in range(0,(len(train_eule_data)- (2*anticipation_size))):
# 		#Get euler, gyro, and alfa one by one
# 		nowTime = i + anticipation_size
# 		velocity_onedata = velocity_o[i].reshape(1,-1)
# 		
# 		euler_pred_onedata = solve_discontinuity(train_eule_data[i].reshape(1,-1))
# 		gyro_pred_onedata = train_gyro_data[i].reshape(1,-1)
# #		alfa_pred_onedata = train_alfa_data[i].reshape(1,-1)
# 		timestamp_rt[:-1] =timestamp_rt[1:]
# 		timestamp_rt[-1] = train_time_data[i]
# 		gyro_rt[:-1] =gyro_rt[1:]
# 		gyro_rt[-1] = train_gyro_data[i]
# 		if (i==0):
# 			alfa_pred_onedata = np.array(np.zeros(shape=(1, 3), dtype=np.float))
# 		else:
# 			alfa_pred_onedata = (np.diff(gyro_rt,axis=0)/np.diff(timestamp_rt, axis=0).reshape(-1, 1))*np.pi/180

# 		timestamp_plot_onedata = i/60 #timestamp_plot[i]

# ####################################Without SG Filter
# 		# Gather data until minimum delay is fulfilled
# 		if rt_counter < DELAY_SIZE:
# 			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
# #			temp_rt = tempNorm.transform(temp_rt)
# 			x_seq_rt[:-1] = x_seq_rt[1:]
# 			x_seq_rt[-1] = temp_rt
# 			rt_counter += 1
# #			if (rt_counter ==sliding_window_size):
# #				seq_df = pd.DataFrame(x_seq_rt)
# #				seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
# #				temp_seq_rt = np.array(seq_df)
# #				xd_seq_rt[:-1] = xd_seq_rt[1:]
# #				xd_seq_rt[-1] = temp_seq_rt[-1]
# #			else:
# #				xd_seq_rt[:-1] = xd_seq_rt[1:]
# #				xd_seq_rt[-1] = temp_rt
# 			continue			
# 		else:
# 			#Get temp cap crp and nop
# 			tempCap = cap_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, alfa_pred_onedata, anticipation_time)
# 			tempCrp = crp_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, anticipation_time)
# 			tempNop = nop_prediction(euler_pred_onedata, anticipation_time)
# 			
# 			cap_pred_rt = np.concatenate((cap_pred_rt, tempCap), axis = 0)
# 			crp_pred_rt = np.concatenate((crp_pred_rt, tempCrp), axis = 0)
# 			nop_pred_rt = np.concatenate((nop_pred_rt, tempNop), axis = 0)

# 			xdd_seq_rt = tempNorm.transform(x_seq_rt)
# 			y_sample = new_sess.run(y, feed_dict={x:xdd_seq_rt.reshape(1,DELAY_SIZE,input_nm)})

# ##	        Switch to NOP if velocity under 8deg/s
# #			for j in range(0,3):
# #				if (np.abs(velocity_onedata[:,j])<1.55):
# #					y_sample[:,j] = tempNop[:,j]

#             #SoftSwitching
# 			for j in range(0,3):
# 				xin = (np.abs(velocity_onedata[:,j])-midVel[j])/avgVel[j]
# 				alfa = sigmoid(xin)
# 				y_sample[:,j] = alfa*y_sample[:,j] + (1-alfa)*tempNop[:,j]

# 			ann_pred_rt = np.concatenate((ann_pred_rt, y_sample), axis =0)
# 			
# 			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
# #			temp_rt = tempNorm.transform(temp_rt)
# 			x_seq_rt[:-1] = x_seq_rt[1:]
# 			x_seq_rt[-1] = temp_rt

# #			seq_df = pd.DataFrame(x_seq_rt)
# #			seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
# #			temp_seq_rt = np.array(seq_df)
# #			xd_seq_rt[:-1] = xd_seq_rt[1:]
# #			xd_seq_rt[-1] = temp_seq_rt[-1]

# #
# 		if graphShow == 'movement':
# 			if coordinate == 'pitch':
# 				y_plot[0] = train_eule_data[nowTime][0]
# 				#CRP
# 				y_plot[1] = tempCrp[0][0]
# 				#CAP
# 				y_plot[2] = tempCap[0][0]
# 				#NOP
# 				y_plot[3] = tempNop[0][0]
# 				#ANN
# 				y_plot[4] = y_sample[0][0]
# 			elif coordinate == 'roll':
# 				y_plot[0] = train_eule_data[nowTime][1]
# 				#CRP
# 				y_plot[1] = tempCrp[0][1]
# 				#CAP
# 				y_plot[2] = tempCap[0][1]
# 				#NOP
# 				y_plot[3] = tempNop[0][1]
# 				#ANN
# 				y_plot[4] = y_sample[0][1]
# 			elif coordinate == 'yaw':
# 				y_plot[0] = train_eule_data[nowTime][2]
# 				#CRP
# 				y_plot[1] = tempCrp[0][2]
# 				#CAP
# 				y_plot[2] = tempCap[0][2]
# 				#NOP
# 				y_plot[3] = tempNop[0][2]
# 				#ANN
# 				y_plot[4] = y_sample[0][2]
# 		rtp.RealTimePlot(float(timestamp_plot_onedata), y_plot)

# # #####################################With SG Filter
# #		# Gather data until minimum delay is fulfilled
# #		if rt_counter < DELAY_SIZE:
# #			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
# #			temp_rt = tempNorm.transform(temp_rt)
# #			x_seq_rt[:-1] = x_seq_rt[1:]
# #			x_seq_rt[-1] = temp_rt
# #			rt_counter += 1
# #			continue			
# #		else:
# #			y_NN = new_sess.run(y, feed_dict={x:x_seq_rt.reshape(1,DELAY_SIZE,input_nm)})
# #			ann_dummy_rt = np.concatenate((ann_dummy_rt, y_NN), axis =0)
# #			ann_pred_rt = np.concatenate((ann_pred_rt, y_NN), axis =0)
# #			
# #			#Get temp cap crp and nop
# #			tempCap = cap_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, alfa_pred_onedata, anticipation_time)
# #			tempCrp = crp_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, anticipation_time)
# #			tempNop = nop_prediction(euler_pred_onedata, anticipation_time)
# #			
# #			cap_pred_rt = np.concatenate((cap_pred_rt, tempCap), axis = 0)
# #			crp_pred_rt = np.concatenate((crp_pred_rt, tempCrp), axis = 0)
# #			nop_pred_rt = np.concatenate((nop_pred_rt, tempNop), axis = 0)
# #			
# #			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
# #			temp_rt = tempNorm.transform(temp_rt)
# #			x_seq_rt[:-1] = x_seq_rt[1:]
# #			x_seq_rt[-1] = temp_rt
# #	
# #			if (m == 0):
# #				#Array less than ideal input(Sg sliding windows and end buff sliding)
# #				if (k<SG_window_size+buff_window):
# #					k = k+1
# #					continue
# #				else:
# #					#Perform savgol only without optimal buffer
# #					temp = signal.savgol_filter(ann_dummy_rt[0:k,:], SG_window_size, 3, axis=0)
# #					ann_pred_rt[0:SG_window_size,:] = temp[0:SG_window_size,:]
# #					m = m+1
# #					k = 0
# #			elif(i<(len(euler_o)-anticipation_size-1)):
# #				#Array less than windows size(sg sliding windows)
# #				if (k<SG_window_size):
# #					k = k+1
# #				else:
# #					#If necesarry input is fulfilled
# #					temp = signal.savgol_filter(ann_dummy_rt[m*SG_window_size-buff_window:m*SG_window_size + k + buff_window,:], SG_window_size, 3, axis=0)
# #					ann_pred_rt[m*SG_window_size:m*SG_window_size+k,:] = temp[buff_window:-buff_window,:]
# #					m = m+1
# #					k = 0
# #			else:
# #				makst = math.floor((len(euler_o)-anticipation_size)/SG_window_size)
# #				k = SG_window_size
# #				for t in range(m,makst):
# #					if (t < makst-1):
# #						temp = signal.savgol_filter(ann_dummy_rt[t*SG_window_size-buff_window:t*SG_window_size + k + buff_window,:], SG_window_size, 3, axis=0)
# #						ann_pred_rt[t*SG_window_size:t*SG_window_size+k,:] = temp[buff_window:-buff_window,:]
# #					else:
# #						temp = signal.savgol_filter(ann_dummy_rt[t*SG_window_size-buff_window::,:], SG_window_size, 3, axis=0)
# #						ann_pred_rt[t*SG_window_size::,:] = temp[buff_window::,:]
# # 				
# ## 		#Plotting realtime
# #		y_sample = ann_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
# #		tempCap = cap_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
# #		tempCrp = crp_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
# #		tempNop = nop_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
# #		tempActual = train_eule_data[nowTime-(SG_window_size + buff_window)].reshape(1,-1)
# #		
# #		if graphShow == 'movement':
# #			if coordinate == 'pitch':
# #				y_plot[0] = tempActual[0][0]
# #				#CRP
# #				y_plot[1] = tempCrp[0][0]
# #				#CAP
# #				y_plot[2] = tempCap[0][0]
# #				#NOP
# #				y_plot[3] = tempNop[0][0]
# #				#ANN
# #				y_plot[4] = y_sample[0][0]
# #			elif coordinate == 'roll':
# #				y_plot[0] = tempActual[0][1]
# #				#CRP
# #				y_plot[1] = tempCrp[0][1]
# #				#CAP
# #				y_plot[2] = tempCap[0][1]
# #				#NOP
# #				y_plot[3] = tempNop[0][1]
# #				#ANN
# #				y_plot[4] = y_sample[0][1]
# #			elif coordinate == 'yaw':
# #				y_plot[0] = tempActual[0][2]
# #				#CRP
# #				y_plot[1] = tempCrp[0][2]
# #				#CAP
# #				y_plot[2] = tempCap[0][2]
# #				#NOP
# #				y_plot[3] = tempNop[0][2]
# #				#ANN
# #				y_plot[4] = y_sample[0][2]
# #		rtp.RealTimePlot(float(timestamp_plot_onedata), y_plot)


# """Calculate online error"""
# #Error is defined as difference between predicted head orientation. 
# #Actual head orientation = Current head orientation shifted by s time
# euler_ann_err_rt = np.abs(ann_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
# euler_cap_err_rt = np.abs(cap_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
# euler_crp_err_rt = np.abs(crp_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
# euler_nop_err_rt = np.abs(nop_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])


# # Calculate average error
# """offline"""

# # Split error value
# euler_ann_err_train = euler_ann_err[: int(TRAIN_SIZE*data_length)] 
# euler_cap_err_train = euler_cap_err[: int(TRAIN_SIZE*data_length)]
# euler_crp_err_train = euler_crp_err[: int(TRAIN_SIZE*data_length)]
# euler_nop_err_train = euler_nop_err[: int(TRAIN_SIZE*data_length)]

# euler_ann_err_test = euler_ann_err[int((1-TEST_SIZE)*data_length):] 
# euler_cap_err_test = euler_cap_err[int((1-TEST_SIZE)*data_length):] 
# euler_crp_err_test = euler_crp_err[int((1-TEST_SIZE)*data_length):] 
# euler_nop_err_test = euler_nop_err[int((1-TEST_SIZE)*data_length):]

# ann_mae = np.nanmean(np.abs(euler_ann_err), axis=0)
# cap_mae = np.nanmean(np.abs(euler_cap_err), axis=0)
# crp_mae = np.nanmean(np.abs(euler_crp_err), axis=0)
# nop_mae = np.nanmean(np.abs(euler_nop_err), axis=0)

# ann_mae_train = np.nanmean(np.abs(euler_ann_err_train), axis=0)
# cap_mae_train = np.nanmean(np.abs(euler_cap_err_train), axis=0)
# crp_mae_train = np.nanmean(np.abs(euler_crp_err_train), axis=0)
# nop_mae_train = np.nanmean(np.abs(euler_nop_err_train), axis=0)

# ann_mae_test = np.nanmean(np.abs(euler_ann_err_test), axis=0)
# cap_mae_test = np.nanmean(np.abs(euler_cap_err_test), axis=0)
# crp_mae_test = np.nanmean(np.abs(euler_crp_err_test), axis=0)
# nop_mae_test = np.nanmean(np.abs(euler_nop_err_test), axis=0)

# # Calculate max error
# ann_max = np.nanmax(np.abs(euler_ann_err), axis=0)
# cap_max = np.nanmax(np.abs(euler_cap_err), axis=0)
# crp_max = np.nanmax(np.abs(euler_crp_err), axis=0)
# nop_max = np.nanmax(np.abs(euler_nop_err), axis=0)

# # Calculate 99% Percentile
# final_ann_99 = np.nanpercentile(euler_ann_err,99, axis = 0)
# final_cap_99 = np.nanpercentile(euler_cap_err,99, axis = 0)
# final_crp_99 = np.nanpercentile(euler_crp_err,99, axis = 0)
# final_nop_99 = np.nanpercentile(euler_nop_err,99, axis = 0)

# final_ann_99_test = np.nanpercentile(euler_ann_err_test,99, axis = 0)
# final_cap_99_test = np.nanpercentile(euler_cap_err_test,99, axis = 0)
# final_crp_99_test = np.nanpercentile(euler_crp_err_test,99, axis = 0)
# final_nop_99_test = np.nanpercentile(euler_nop_err_test,99, axis = 0)

# # get rms stream
# ann_rms_stream = np.apply_along_axis(rms,1,euler_ann_err)
# cap_rms_stream = np.apply_along_axis(rms,1,euler_cap_err)
# crp_rms_stream = np.apply_along_axis(rms,1,euler_crp_err)
# nop_rms_stream = np.apply_along_axis(rms,1,euler_nop_err)


# # calculate error rms mean
# ann_rms = np.nanmean(ann_rms_stream)
# cap_rms = np.nanmean(cap_rms_stream)
# crp_rms = np.nanmean(crp_rms_stream)
# nop_rms = np.nanmean(nop_rms_stream)


# """online"""
# #Split error value online
# euler_ann_err_train_rt = euler_ann_err_rt[: int(TRAIN_SIZE*len(euler_ann_err_rt))] 
# euler_cap_err_train_rt = euler_cap_err_rt[: int(TRAIN_SIZE*data_length)]
# euler_crp_err_train_rt = euler_crp_err_rt[: int(TRAIN_SIZE*data_length)]
# euler_nop_err_train_rt = euler_nop_err_rt[: int(TRAIN_SIZE*data_length)]

# euler_ann_err_test_rt = euler_ann_err_rt[int((1-TEST_SIZE)*len(euler_ann_err_rt)):] 
# euler_cap_err_test_rt = euler_cap_err_rt[int((1-TEST_SIZE)*data_length):] 
# euler_crp_err_test_rt = euler_crp_err_rt[int((1-TEST_SIZE)*data_length):] 
# euler_nop_err_test_rt = euler_nop_err_rt[int((1-TEST_SIZE)*data_length):]

# ann_mae_rt = np.nanmean(np.abs(euler_ann_err_rt), axis=0)
# cap_mae_rt = np.nanmean(np.abs(euler_cap_err_rt), axis=0)
# crp_mae_rt = np.nanmean(np.abs(euler_crp_err_rt), axis=0)
# nop_mae_rt = np.nanmean(np.abs(euler_nop_err_rt), axis=0)

# ann_mae_train_rt = np.nanmean(np.abs(euler_ann_err_train_rt), axis=0)
# cap_mae_train_rt = np.nanmean(np.abs(euler_cap_err_train_rt), axis=0)
# crp_mae_train_rt = np.nanmean(np.abs(euler_crp_err_train_rt), axis=0)
# nop_mae_train_rt = np.nanmean(np.abs(euler_nop_err_train_rt), axis=0)

# ann_mae_test_rt = np.nanmean(np.abs(euler_ann_err_test_rt), axis=0)
# cap_mae_test_rt = np.nanmean(np.abs(euler_cap_err_test_rt), axis=0)
# crp_mae_test_rt = np.nanmean(np.abs(euler_crp_err_test_rt), axis=0)
# nop_mae_test_rt = np.nanmean(np.abs(euler_nop_err_test_rt), axis=0)

# #Max
# ann_max_rt = np.nanmax(np.abs(euler_ann_err_rt), axis=0)
# cap_max_rt = np.nanmax(np.abs(euler_cap_err_rt), axis=0)
# crp_max_rt = np.nanmax(np.abs(euler_crp_err_rt), axis=0)
# nop_max_rt = np.nanmax(np.abs(euler_nop_err_rt), axis=0)

# #99MAE
# final_ann_rt_99 = np.nanpercentile(euler_ann_err_rt,99, axis = 0)
# final_cap_rt_99 = np.nanpercentile(euler_cap_err_rt,99, axis = 0)
# final_crp_rt_99 = np.nanpercentile(euler_crp_err_rt,99, axis = 0)
# final_nop_rt_99 = np.nanpercentile(euler_nop_err_rt,99, axis = 0)

# final_ann_rt_99_test = np.nanpercentile(euler_ann_err_test_rt,99, axis = 0)
# final_cap_rt_99_test = np.nanpercentile(euler_cap_err_test_rt,99, axis = 0)
# final_crp_rt_99_test = np.nanpercentile(euler_crp_err_test_rt,99, axis = 0)
# final_nop_rt_99_test = np.nanpercentile(euler_nop_err_test_rt,99, axis = 0)

# ann_rms_stream_rt = np.apply_along_axis(rms,1,euler_ann_err_rt)
# cap_rms_stream_rt = np.apply_along_axis(rms,1,euler_cap_err_rt)
# crp_rms_stream_rt = np.apply_along_axis(rms,1,euler_crp_err_rt)
# nop_rms_stream_rt = np.apply_along_axis(rms,1,euler_nop_err_rt)
# #
# print('Offline - MAE')
# print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
# print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
# print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
# print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))

# print('Online - MAE')
# print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_rt[0], ann_mae_rt[1], ann_mae_rt[2]))
# print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(cap_mae_rt[0], cap_mae_rt[1], cap_mae_rt[2]))
# print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(crp_mae_rt[0], crp_mae_rt[1], crp_mae_rt[2]))
# print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(nop_mae_rt[0], nop_mae_rt[1], nop_mae_rt[2]))

# print('\nOffline - 99% MAE')
# print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_99[0], final_ann_99[1], final_ann_99[2]))
# print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_cap_99[0], final_cap_99[1], final_cap_99[2]))
# print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_crp_99[0], final_crp_99[1], final_crp_99[2]))
# print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_nop_99[0], final_nop_99[1], final_nop_99[2]))

# print('Online - 99% MAE')
# print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
# print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
# print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
# print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))