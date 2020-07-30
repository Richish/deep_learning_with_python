#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:05:31 2020

@author: unravel
"""


#%%
import pandas as pd
import numpy as np

import os
from keras import layers, models, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
dataset_dir=os.path.join(os.getcwd(), "/Users/unravel/learnings/MachineLearning/DataSets")
base_dir=os.path.join(dataset_dir, "chest_xray_pneumonia_normal_kaggle_5863")
train_dir=os.path.join(base_dir, "train")
val_dir=os.path.join(base_dir, "val")
validation_dir=val_dir
test_dir=os.path.join(base_dir, "test")
train_normal_dir=os.path.join(train_dir,"normal")
#%%
conv_base= VGG16(weights="imagenet", include_top=False, input_shape=(350,350,3))
conv_base.summary()

#%%
datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

def train_generator(directory, sample_count):
  features=np.zeros(shape=(sample_count, 10,10, 512))
  labels=np.zeros(shape=(sample_count))
  generator=datagen.flow_from_directory(directory=directory, target_size=(350,350), class_mode='binary', batch_size=batch_size)
  i=0
  for input_batch, labels_batch in generator:
    features_batch=conv_base.predict(x=input_batch, verbose=1)
    features[i*batch_size:(i+1)*batch_size]=features_batch
    labels[i*batch_size:(i+1)*batch_size]=labels_batch
    i+=1
    if i*batch_size>=sample_count:
      break
  return features, labels

def val_generator(directory, sample_count):
  features=np.zeros(shape=(sample_count, 10,10, 512))
  labels=np.zeros(shape=(sample_count))
  generator=datagen.flow_from_directory(directory=directory, target_size=(350,350), class_mode='binary', batch_size=3)
  i=0
  for input_batch, labels_batch in generator:
    features_batch=conv_base.predict(x=input_batch, verbose=1)
    features[i*batch_size:(i+1)*batch_size]=features_batch
    labels[i*batch_size:(i+1)*batch_size]=labels_batch
    i+=1
    if i*batch_size>=sample_count:
      break
  return features, labels

#%%
train_features, train_labels= train_generator(train_dir, 5000)
#validation_features, validation_labels= train_generator(validation_dir, 16)
test_features, test_labels = train_generator(test_dir,620)

train_features=np.reshape(train_features, (5000, (10*10*512)))
#validation_features=np.reshape(validation_features, (16, (4*4*512)))
test_features=np.reshape(test_features, (620, (10*10*512)))

#%%
from keras import models, layers, optimizers
model=models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=10*10*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

#%%
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
#%%
history=model.fit(train_features, train_labels, epochs=15, batch_size=20, validation_data=(test_features, test_labels))

#%%
# plotting the results with learned features
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show(

