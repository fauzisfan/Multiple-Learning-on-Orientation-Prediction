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
from calc_future_orientation import calc_future_orientation

def cap_prediction(euler_o, gyro_o, alfa_o, anticipation_time):
    # Get traveled distance
    theta = np.linalg.norm(
            (gyro_o * anticipation_time/1e3 + (1/2 * alfa_o * (np.square(anticipation_time/1e3)))), axis=1).reshape(-1, 1)
    
    # Get the rotation axis where rotation happens
    angle_axis = gyro_o/np.linalg.norm(gyro_o, axis=1).reshape(-1, 1)
    
    # Calculate predicted orientation
    return calc_future_orientation(euler_o, theta, angle_axis)