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
from keras_preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, #channel_shift_range=0.2,
    fill_mode="nearest", horizontal_flip=False#, vertical_flip=True
)

#%%
# visualizing augmentation of 1 image
from keras.preprocessing import image
import matplotlib.pyplot as plt
# read 4th image
normal_images_in_train=[os.path.join(train_normal_dir,fname) for fname in os.listdir(train_normal_dir)]
img_path=normal_images_in_train[3]
img=image.load_img(path=img_path, target_size=(150,150))

x=image.img_to_array(img=img)
x=x.reshape((1,)+x.shape)
print(x.shape)

i=0
for batch in datagen.flow(x=x, batch_size=1):
  plt.figure(i)
  img_plot=plt.imshow(image.array_to_img(x=batch[0]))
  i+=1
  if i%4==0:
    break

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

model.add(layers.Dropout(rate=0.5))# dropout layer added to module

model.add(layers.Dense(units=512, activation="relu" ))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss="binary_crossentropy", metrics=["acc"])

#%%
train_datagen=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest', horizontal_flip=False,
rescale=1./255
)
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(directory=train_dir, target_size=(150,150), class_mode='binary', batch_size=50)
validation_generator=validation_datagen.flow_from_directory(directory=validation_dir, target_size=(150,150), class_mode='binary', batch_size=1)

#%%
history=model.fit_generator(generator=train_generator, steps_per_epoch=110,\
                            epochs=30, validation_data=validation_generator, \
                                validation_steps=16)
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
