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
conv_base= VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv_base.summary()

# adding dense layers on top of conv base

from keras import models, layers, optimizers

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

print("Number of trainable features of conv base before freezing: ", len(model.trainable_weights), len(model.non_trainable_weights))
conv_base.trainable=False
print("Number of trainable features of conv base after freezing: ", len(model.trainable_weights), len(model.non_trainable_weights))

# compile the model
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])

#%%
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen=ImageDataGenerator(rescale=1.0/255.0, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
train_generator=train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=20, class_mode='binary')

test_datagen=ImageDataGenerator(rescale=1.0/255.0)
validation_generator=train_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=20, class_mode='binary')

xhistory = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=30,
validation_data=validation_generator,
validation_steps=50)

