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

def preparets(x, y, delay_size):
    assert 0 < delay_size < x.shape[0]
    x_seq       = np.atleast_3d(np.array([x[start:(start+delay_size)] for start in range(0, x.shape[0]-delay_size)]))
    y_seq       = y[delay_size::]    
    return x_seq, y_seq