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

#%%
dataset_dir=os.path.join(os.getcwd(), "/Users/unravel/learnings/MachineLearning/DataSets")
base_dir=os.path.join(dataset_dir, "chest_xray_pneumonia_normal_kaggle_5863")
train_dir=os.path.join(base_dir, "train")
val_dir=os.path.join(base_dir, "val")
test_dir=os.path.join(base_dir, "test")

#%%
from keras import models, layers, optimizers

model=models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),activation="relu", input_shape=(150,150,3) ))
model.add(layers.MaxPool2D(pool_size=(2,2) ))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3),activation="relu" ))
model.add(layers.MaxPool2D(pool_size=(2,2) ))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3),activation="relu" ))
model.add(layers.MaxPool2D(pool_size=(2,2) ))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3),activation="relu" ))
model.add(layers.MaxPool2D(pool_size=(2,2) ))

model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation="relu" ))
model.add(layers.Dense(units=1, activation="sigmoid"))

#%%
model.summary()
#%%
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss="binary_crossentropy", metrics=["acc"])

#%%
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    directory=train_dir, target_size=(150,150), color_mode="rgb", class_mode="binary", batch_size=20, shuffle=True 
)

validation_generator=validation_datagen.flow_from_directory(
    directory=val_dir, target_size=(150,150), color_mode="rgb", class_mode="binary", batch_size=20, shuffle=True 
)

#%%
# see what is being generated
for data_batch, labels_batch in train_generator:
  print("data batch shape:", data_batch.shape)
  print("labels batch shape:", labels_batch.shape)
  break # printing only 1 batch's value

#%%
history=model.fit_generator(generator=train_generator, steps_per_epoch=100, # steps per epoch is how many steps of generator will be executed in 1 epoch
                            # since 1 step of generator is 20 images, hence 2000 images require 100 steps in 1 epoch for all images
                            epochs=30, validation_data=validation_generator, validation_steps=50
                            )

#%%
model.save("pneumonia_basic_1.h5")
#%%
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
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()

