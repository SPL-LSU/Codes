#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[4]:


tf.keras.version()


# In[5]:


#keras uses layers to build models, models are a graph of layers


# In[8]:


from tensorflow.keras import layers

#type of model: Sequential
model = tf.keras.Sequential()
#creates a "dense" layer with 64 units 
model.add(layers.Dense(64,activation = 'relu'))

model.add(layers.Dense(64, activation = 'relu'))
#output layer has 10 units
model.add(layers.Dense(10))


# In[9]:


#activation: sets activation function, default none
#kernel_initializer/bias_initializer: creates layer's weights (kernels and biases)
    #name or callable, default "Glorot uniform"
#kernel_regularizer/bias_regularizer: creates schemes that apply weights
    #ex L1 or L2, default none


# In[10]:


#CREATES LAYERS

#creates relu layer
layers.Dense(64, activation = 'relu')
#alternatively:
layers.Dense(64, activation = tf.nn.relu)
#l1: lasso regression
#l2: ridge regression

#l1 regularization of factor .01 applied to kernel matrix
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(.01))

#linear layer that makes random orthonoganal matrix
layers.Dense(65, kernel_initializer="orthogonal")

#linear layer with bias vector initialized to 2.0
layers.Dense(64,bias_initializer=tf.keras.initializers.Constant(2.0))


# In[11]:


#SET UP TRAINING

model=tf.keras.Sequential([
    #add in blank layers
    layers.Dense(64,activation='relu',input_shape=(32,)),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)])
#optimizer: specifies training procedure, ex Adam or SGD
#loss: function to minimize during optimization
#metrics: monitors training
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[13]:


#CREATE TRAINING MODEL
#this example uses mean-squared error regression

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',#mean squared error
              metrics=['mae'])#mean absolute error

model.compile(optimizer=tf.keras.optimizers.RMSprop(.01),
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[14]:


#TRAIN

import numpy as np

data = np.random.random((1000,32))
labels = np.random.random((1000,10))

model.fit(data,labels, epochs=10, batch_size=32)
#epochs: an iteration over the entire set of input data
#batch_size: integer size of each batch that will be iterated over
    #last batch may be smaller
#validation_data: displays loss and metrics for data after each epoch


# In[15]:


#TRAINING WITH VALIDATION

import numpy as np

data = np.random.random((1000,32))
labels = np.random.random((1000,10))

val_data = np.random.random((100,32))
val_labels = np.random.random((100,10))

model.fit(data, labels, epochs=10, batch_size=32,
         validation_data=(val_data,val_labels))


# In[16]:


#TRAIN FROM DATA SETS

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.fit(dataset, epochs=10)


# In[18]:


#validation with datasets

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10,
         validation_data=val_dataset)


# In[19]:


#EVALUATE

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size = 32)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.evaluate(dataset)


# In[20]:


#PREDICT OUTPUT
result = model.predict(data, batch_size = 32)
print(result.shape)


# In[ ]:




