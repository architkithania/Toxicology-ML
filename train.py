#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import os

import numpy as np
import tensorflow as tf
import h5py


# In[2]:


def generate_data(data, labels):
    with open(os.path.join(os.path.curdir, data), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        graphs = [row for row in reader]
        
    X = np.array(graphs, dtype=int)
    data_points = X.shape[0] // X.shape[1]

    X = X.reshape((data_points, X.shape[1], X.shape[1], 1))
    y = np.genfromtxt(labels, skip_header=1, usecols=1, dtype=int)
    
    idx = np.random.permutation(y.shape[0])
    X_train, y_train = X[idx], y[idx]

    y_hot = np.zeros((y_train.shape[0], 2))
    y_hot[np.arange(y_train.shape[0]), y_train] = 1
    return X_train, y_hot


# In[9]:


get_ipython().run_line_magic('pinfo2', 'tf.keras.layers.Conv2D')


# In[15]:


def create_modal():
    modal = tf.keras.Sequential()
    modal.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:], padding='same'))
    modal.add(tf.keras.layers.BatchNormalization())
    modal.add(tf.keras.layers.Dropout(0.3))
    
    modal.add(tf.keras.layers.Conv2D(64, (6, 6), activation='relu', input_shape=X.shape[1:], padding='same'))
    modal.add(tf.keras.layers.BatchNormalization())
    modal.add(tf.keras.layers.MaxPooling2D((3, 3)))
    modal.add(tf.keras.layers.Dropout(0.3))
    
    modal.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', input_shape=X.shape[1:], padding='same'))
    modal.add(tf.keras.layers.BatchNormalization())
    modal.add(tf.keras.layers.MaxPooling2D((3, 3)))
    modal.add(tf.keras.layers.Dropout(0.3))
    
    modal.add(tf.keras.layers.Conv2D(256, (1, 1), activation='relu', input_shape=X.shape[1:], padding='same'))
    modal.add(tf.keras.layers.BatchNormalization())
    modal.add(tf.keras.layers.MaxPooling2D((3, 3)))
    modal.add(tf.keras.layers.Dropout(0.3))
    
    modal.add(tf.keras.layers.Flatten())
    modal.add(tf.keras.layers.Dense(256, activation='relu'))
    modal.add(tf.keras.layers.Dense(2, activation='softmax'))
    return modal


# In[11]:


X, y = generate_data('train_data/train_graphs.csv', 'train_data/train_labels.csv')


# In[5]:


neg = y[y[np.arange(y.shape[0]), 0] == 1].shape[0]
pos = y[y[np.arange(y.shape[0]), 1] == 1].shape[0]

total = y.shape[0]

class_weights = {
    0: (1 / neg)*(total)/2.0,
    1: (1 / pos)*(total)/2.0
}


# In[20]:


model = create_modal()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=5, class_weights=class_weights)


# In[22]:


model.save('tox-model.h5')

