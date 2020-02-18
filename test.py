#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import tensorflow as tf


# In[9]:


def generate_data(data):
    X = np.genfromtxt(data, delimiter=',')
    data_points = X.shape[0] // X.shape[1]

    X = X.reshape((data_points, X.shape[1], X.shape[1], 1))

    return X


# In[10]:


def get_predictions(modal, data):
    output = modal.predict(X_test)
    predictions = np.zeros(output.shape)
    
    predictions[np.arange(predictions.shape[0]), output.argmax(axis=1)] = 1
    return predictions


# In[11]:


def create_output_file(predictions):
    value_vector = predictions.astype(str)
    with open('labels.txt', 'w+') as f:
        for val in value_vector:
            f.write(val + '\n')


# In[12]:


model = tf.keras.models.load_model('tox-model.h5')


# In[13]:


X_test = generate_data('../Tox21_AR/score_graphs.csv')


# In[14]:


output = model.predict_classes(X_test, 2)


# In[15]:


create_output_file(output)

