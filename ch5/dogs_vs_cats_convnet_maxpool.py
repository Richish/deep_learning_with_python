# -*- coding: utf-8 -*-
"""
"""
#%%

import os
from keras import layers, models
#%%
data_dir=os.path.join(os.getcwd(), "../../DataSets/DogsVsCats/")
print(os.listdir(data_dir))
src_train=os.path.join(data_dir, "train")
src_test=os.path.join(data_dir, "test1")

src_train_images=os.listdir(src_train)
src_test_images=os.listdir(src_test)

print(src_train_images[0:10])
#%% create target dirs

target_dir=os.path.join(os.getcwd(),"../data")
print(os.listdir(target_dir))

target_train_dir=os.path.join(target_dir, "train")
target_validation_dir=os.path.join(target_dir, "validation")

target_test_dir=os.path.join(target_dir, "test")

os.mkdir(target_train_dir)
os.mkdir(target_validation_dir)
os.mkdir(target_test_dir)


#%%
import os, shutil
original_dataset_dir = os.path.join(os.getcwd(), "../DataSets/DogsVsCats/")
src_train=os.path.join(original_dataset_dir, "train")
src_test=os.path.join(original_dataset_dir, "test1")

base_dir = os.path.join(os.getcwd(),"data/cats_and_dogs_small")
#os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)
#%%
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(src_train, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
#%%
import os, shutil
original_dataset_dir = os.path.join(os.getcwd(), "../DataSets/DogsVsCats/")
src_train=os.path.join(original_dataset_dir, "train")
src_test=os.path.join(original_dataset_dir, "test1")

base_dir = os.path.join(os.getcwd(),"../data/cats_and_dogs_small")
#os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
print(os.listdir(test_cats_dir))

#%% keras image generators
from keras_preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(directory=train_dir, target_size=(150,150), color_mode='rgb', class_mode='binary', batch_size=20)

validation_generator=test_datagen.flow_from_directory(directory=validation_dir, target_size=(150,150), color_mode='rgb', class_mode='binary', batch_size=20)

#test_generator=test_datagen.flow_from_directory(directory=test_dir, target_size=(150,150), color_mode='rgb', class_mode='binary', batch_size=20)

#%% init model
from keras import models, layers
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), input_shape=(150,150,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.summary()

#%% compile model
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

#%% fit model
history=model.fit_generator(train_generator, epochs=3, steps_per_epoch=100, validation_data=validation_generator, validation_steps=50)
#%%  save model
model.save('dogs_and_cats_small_1.h5')

#%% plot history
model_h=models.load_model('dogs_and_cats_small_1.h5')
print(model_h.state_updates)
#%%
import matplotlib.pyplot as plt

hist_dict=history.history
acc=hist_dict['acc']
loss=hist_dict['loss']
val_acc=hist_dict['val_acc']
val_loss=hist_dict['val_loss']
epochs=range(1, len(acc)+1)
plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs,val_acc,'ro', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'b', label='Training loss')
plt.plot(epochs,val_loss,'r', label='Validation loss')
plt.legend()
#plt.figure()
plt.show()

#%% using vg166
from keras.applications import VGG16
conv_base= VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))

#%%
conv_base.summary()