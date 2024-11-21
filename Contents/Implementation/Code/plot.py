import pandas as pd
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense, LSTM,Conv2D,AveragePooling1D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import func4 as fn
import tensorflow as tf
import os

# python libs
from sys import version
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from numpy import argmax

# local libs
import cfg #importing local library cfg

from selfAttn import SelfAttn

from numpy import array
from keras.models import Sequential,Model
from keras.layers import LSTM,Conv1D,MaxPooling1D,Flatten,Input,UpSampling1D,Reshape
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed#define input sequence
from keras.layers import concatenate
from sklearn.metrics import confusion_matrix
import os
# os.chdir(os.path.join(os.getcwd(), '..'))
# os.getcwd()

from keras.layers import concatenate
from src.utils import get_dataset
from src.utils import select_data

import scipy.io as sio
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.utils import to_categorical

from keras.preprocessing import sequence

from keras.models import Sequential
from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Concatenate



# input_vec = Input(shape=(180, 60))  # adjust this to match your vector data
#
# x = Conv1D(16, 3, activation='relu', padding='same')(input_vec)
# x = MaxPooling1D(2, padding='same')(x)
# encoded = Conv1D(8, 3, activation='relu', padding='same')(x)
#
# x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
# x = UpSampling1D(2)(x)
# decoded = Conv1D(60, 3, activation='sigmoid', padding='same')(x)
#
# autoencoder = Model(input_vec, decoded)



from keras.models import model_from_json

# load json and create model
json_file = open('ae_skel_18_9_23.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
modelAE = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[1].output)


model = Sequential(name='conv_LSTM2')
model.add(Conv1D(16, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform',
                 input_shape=(180, 6)))
model.add(Conv1D(32, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model.add(Conv1D(128, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
model.add(LSTM(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

model2 = Sequential(name='conv_LSTM2')
model2.add(Conv1D(16, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform',
                  input_shape=(180, 6)))
model2.add(Conv1D(32, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(64, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model2.add(Conv1D(128, 3, padding='same', activation='relu', strides=1, kernel_initializer='glorot_uniform'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
model2.add(LSTM(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

merged_output = Concatenate()([model.output, model2.output])

merged_output = LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, name='lstm1')(merged_output)
merged_output = LSTM(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, name='lstm2')(merged_output)
mer = Flatten()(merged_output)
fully_connected = Dense(27, activation='softmax')(mer)
combined_model = Model([model1.input, model2.input], fully_connected)


from keras.utils.vis_utils import plot_model
plot_model(combined_model, to_file='54.png', show_shapes=True, show_layer_names=True)