from __future__ import print_function, division

from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, MaxPooling2D, Lambda
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from scipy.signal import detrend
import matplotlib.pyplot as plt
from tensorflow.keras.losses import mean_squared_error
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
Ka = tf.keras
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
# Common imports
import numpy as np 
import os
import  wiggle
from tensorflow import keras
import datetime
from spela.spectrogram import Spectrogram
import pdb
import librosa

model = Sequential()
input_shape = (1, 16000)

model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(input_shape),
                      return_decibel_spectrogram=False, power_spectrogram=2.0,
                      trainable_kernel=False, name='static_stft'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=mean_squared_error, metrics=["acc"])
print(model.summary())

x = np.random.random(16000)
x = np.reshape(x,(1,1,16000))
model.predict(x)