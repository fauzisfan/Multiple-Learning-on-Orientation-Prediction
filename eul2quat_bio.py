import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers

def eul2quat_bio (eul):
    # Convert Euler [oculus] to Quaternion [oculus]
    eul = np.deg2rad(eul)
    X_eul = eul[:,0]
    Y_eul = eul[:,1]
    Z_eul = eul[:,2]

    cos_pitch, sin_pitch = np.cos(X_eul/2), np.sin(X_eul/2)
    cos_yaw, sin_yaw = np.cos(Y_eul/2), np.sin(Y_eul/2)
    cos_roll, sin_roll = np.cos(Z_eul/2), np.sin(Z_eul/2)

    # order: w,x,y,z
    # quat = unit_q(quat)
    quat = np.nan * np.ones( (eul.shape[0],4) )
    quat[:,0] = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
    quat[:,1] = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
    quat[:,2] = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll 
    quat[:,3] = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll     
    return (quat)