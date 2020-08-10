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

def quat2eul_bio(quat):
    temp_zx = 2 * (quat[:,1] * quat[:,3] - quat[:,0] * quat[:,2])
    temp_yx = 2 * (quat[:,1] * quat[:,2] + quat[:,0] * quat[:,3])
    temp_zy = 2 * (quat[:,2] * quat[:,3] + quat[:,0] * quat[:,1])
    temp_1 = 2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2])
    temp_2 = (quat[:,0]*quat[:,0]+quat[:,1]*quat[:,1]-quat[:,2]*quat[:,2]-quat[:,3]*quat[:,3])
    temp_3 = 2*(quat[:,0]*quat[:,1]+quat[:,3]*quat[:,2])
    temp_4 = (quat[:,0]*quat[:,0]-quat[:,1]*quat[:,1]-quat[:,2]*quat[:,2]+quat[:,3]*quat[:,3])

    Y_eul = -np.arcsin(temp_zx)
    X_eul = np.arctan2(temp_1, temp_2)
    Z_eul = np.arctan2(temp_3, temp_4)

    eul = np.rad2deg(np.column_stack((X_eul,Y_eul,Z_eul)))

    return (eul)