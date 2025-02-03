#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')


# In[3]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras import regularizers
from sklearn.model_selection import train_test_split


# In[5]:


training_dir="C:/Users/windows/Desktop/past semesters/ML/project/dataset/train"
validation_dir="C:/Users/windows/Desktop/past semesters/ML/project/dataset/val"

def prep_data(augmented,batch_size=32):      
    if augmented:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
        validation_datagen = ImageDataGenerator(rescale=1./255)    

    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # training set
    train_set = train_datagen.flow_from_directory(
        training_dir,
        target_size=(180, 180),  
        batch_size=batch_size,
        class_mode="sparse")
         
    
    validation_set = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(180, 180),
        batch_size=batch_size,  
        class_mode="sparse")
             
    return train_set , validation_set


# In[6]:


def visualize(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].plot(epochs, acc, 'r', label='Training acc')
    axs[0].plot(epochs, val_acc, 'b', label='Validation acc')
    axs[0].set_title('Training and validation accuracy')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(epochs, loss, 'r', label='Training loss')
    axs[1].plot(epochs, val_loss, 'b', label='Validation loss')
    axs[1].set_title('Training and validation loss')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()


# In[8]:


train_dir = training_dir
validation_dir = validation_dir

train_class_counts = {}
for folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        train_class_counts[folder] = num_images

validation_class_counts = {}
for folder in os.listdir(validation_dir):
    class_path = os.path.join(validation_dir, folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        validation_class_counts[folder] = num_images


# In[15]:


train_set,validation_set=prep_data(True)

images, labels = next(train_set)

class_names = train_set.class_indices
class_names = {v: k for k, v in class_names.items()} 


# In[16]:


print("img shape: ", images.shape)
print("labels shape: ", labels.shape)


# In[12]:


plt.imshow(images[0]) 


# In[72]:


train_set,validation_set=prep_data(True,batch_size=32)


# In[73]:


from tensorflow.keras.applications import InceptionV3
base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(180, 180, 3))


# In[74]:


for layer in base_model.layers[:-15]:
    layer.trainable = False


# In[75]:


model = keras.models.Sequential()
model.add(base_model)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(75, activation='relu'))
model.add(keras.layers.Dropout(0.2)) 
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(93, activation='softmax')) 


# In[76]:


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
              metrics=['accuracy'])


# In[77]:


model.summary()


# In[78]:


history = model.fit(train_set,
                    epochs=30,
                    validation_data=validation_set)


# In[79]:


visualize(history)

