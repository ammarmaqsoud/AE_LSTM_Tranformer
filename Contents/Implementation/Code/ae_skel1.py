
import os
os.chdir(os.path.join(os.getcwd(), '..'))
os.getcwd()


from src.utils import get_dataset
from src.utils import select_data



import scipy.io as sio
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from tensorflow.keras.utils import to_categorical

from keras.preprocessing import sequence

from keras.models import Sequential
from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM,Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Concatenate

from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense, LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

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


 

def custom_mse(y_true, y_pred):
    # calculating squared difference between target and predicted values
    loss =  tf.keras.backend.square(y_pred - y_true)  # (batch_size, 2)

    # multiplying the values with weights along batch dimension
    loss = loss * [0.3, 0.7]  # (batch_size, 2)

    # summing both loss values along batch dimension
    loss =  tf.keras.backend.sum(loss, axis=1)  # (batch_size,)

    return loss




#load data
DATA_PATH = os.path.join(os.getcwd(), 'dataoriginal')
os.listdir(DATA_PATH)
depth_path, inertial_path,rgb_path, skeleton_path = get_dataset(DATA_PATH)
activities = list(range(1,28))


def select_subject(d_path, subject):
    select_statement = '_s{}_'.format(subject)
    subjects = []
    for i in d_path:
        if select_statement in i:
            subjects.append(i)
    return subjects


def get_action_number(single_path):
    return int(single_path.split('\\')[-1].split('_')[0][1:])
def get_subject_number(single_path):
    return int(single_path.split('\\')[-1].split('_')[1][1:])
def get_trial_number(single_path):
    return int(single_path.split('\\')[-1].split('_')[2][1:])


# Skel
# train on trial 1,2,3
# test on trial 4
X_train_skel = []
Y_train_skel = []
X_test_skel = []
Y_test_skel = []

resample_len = 180

for path in skeleton_path:
    if get_trial_number(path) in [1,2,3,4]:
        X_train_skel.append(path)
        Y_train_skel.append(get_action_number(path))
    else:
        X_test_skel.append(path)
        Y_test_skel.append(get_action_number(path))

# X_train_skel = [pad_len_inertial(sio.loadmat(x)['d_iner']) for x in X_train_skel]
# X_test_skel = [pad_len_inertial(sio.loadmat(x)['d_iner']) for x in X_test_skel]

X_train_skel = [np.reshape(resample(sio.loadmat(x)['d_skel'], resample_len, axis = -1), (60,180)) for x in X_train_skel]
# X_test_skel = [np.reshape(resample(sio.loadmat(x)['d_skel'], resample_len, axis = -1), (60,180)) for x in X_test_skel]


X_train_skel = np.array(X_train_skel)
# X_test_skel = np.array(X_test_skel)

X_train_skel = np.swapaxes(X_train_skel, 1,2)
# X_test_skel = np.swapaxes(X_test_skel, 1,2)

Y_train_skel = to_categorical(np.array(Y_train_skel) - 1)
# Y_test_skel = to_categorical(np.array(Y_test_skel) - 1)


 
# data normalization  
 

X_train_skel[:,:,3:] = X_train_skel[:,:,3:]/ max(X_train_skel[:,:,3:].max(), abs(X_train_skel[:,:,3:].min()))
X_train_skel[:,:,:3] = X_train_skel[:,:,:3]/ max(X_train_skel[:,:,:3].max(), abs(X_train_skel[:,:,:3].min()))

# X_test_skel[:,:,3:] = X_test_skel[:,:,3:]/ max(X_test_skel[:,:,3:].max(), abs(X_test_skel[:,:,3:].min()))
# X_test_skel[:,:,:3] = X_test_skel[:,:,:3]/ max(X_test_skel[:,:,:3].max(), abs(X_test_skel[:,:,:3].min()))
 



# X_train_skel.shape, Y_train_skel.shape, X_test_skel.shape, Y_test_skel.shape
# print('---------------------------------------------------')
# print('X_train_skel.shape',X_train_skel.shape)
# print('Y_train_skel.shape',Y_train_skel.shape)
# print('X_test_skel.shape',X_test_skel.shape)
# print('Y_test_skel.shape',Y_test_skel.shape)


X_train=X_train_skel
# X_test=X_test_skel


Y_train = Y_train_skel
# Y_test = Y_test_skel
#
# X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
optimizer = Adam(learning_rate=1e-4)

# timesteps=180
# n_features=60
# # define model
# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
# model.add(LSTM(64, activation='relu', return_sequences=False))
# model.add(RepeatVector(timesteps))
# model.add(LSTM(64, activation='relu', return_sequences=True))
# model.add(LSTM(128, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss='mse')
# model.summary()


timesteps=180
n_features=60
# define model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(6, activation='relu', return_sequences=True))
# model.add(RepeatVector(timesteps))

model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X_train, X_train, epochs=200, verbose=1,batch_size=800 )


# modelsaveing1 = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)


model_json = model.to_json()
with open("ae_skel_19_9_23.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("ae_skel_19_9_23.h5")
print("Saved model to disk")


import matplotlib.pyplot as plt

# # Plotting the loss curve
# plt.figure(figsize=[6,4])
# plt.plot(history.history['loss'], 'orange', linewidth=2.0)
# plt.plot(history.history['val_loss'], 'blue', linewidth=2.0)
# plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
# plt.xlabel('Epochs', fontsize=10)
# plt.ylabel('Loss', fontsize=10)
# plt.title('Loss Curves', fontsize=12)

# # Plotting the accuracy curve
# plt.figure(figsize=[6,4])
# plt.plot(history.history['accuracy'], 'orange', linewidth=2.0)
# plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
# plt.xlabel('Epochs', fontsize=10)
# plt.ylabel('Accuracy', fontsize=10)
# plt.title('Accuracy Curves', fontsize=12)
#
# plt.show()



# def show_Test_accuracy():
#     _, accuracy = model.evaluate([X_test[0], X_test[1]], Y_test)
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#
# show_Test_accuracy()



