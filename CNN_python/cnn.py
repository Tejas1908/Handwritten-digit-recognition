import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(8, (3, 3), padding='same',input_shape=(28,28,1),kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Flatten())

# Fully connected layer

model.add(Dense(10,kernel_regularizer=regularizers.l2(0)))

model.add(Activation('softmax'))

model.compile(loss='mean_absolute_error', optimizer=Adam(), metrics=['accuracy'])


model.fit(X_train,Y_train,
          batch_size=128,
          epochs=4,
          verbose=1,
          validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model1')
