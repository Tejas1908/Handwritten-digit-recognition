import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
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

model = load_model('model1')

weights=model.get_weights()

#for i in range(len(weights)):
#    print(weights[i].shape)

conv1=weights[0];

for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(conv1[:,:,0,i],cmap='gray')
    plt.axis('off')
plt.show()

