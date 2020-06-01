#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.tensorflow.org/tutorials/quickstart/beginner


# In[2]:


#use keras to build a neural net that classifies images


# In[3]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


# In[5]:


#loads and prepares MNIST dataset, converts int to float
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


# In[6]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


# In[7]:


#returns a vector of logits, basically unnormalized probabilites
predictions = model(x_train[:1]).numpy()
predictions


# In[9]:


#convert logits to probabilities
tf.nn.softmax(predictions).numpy()


# In[10]:


#takes vector of logits and returns scalar loss for each example
#equal to negative logg probability of the true class
#zero if model is sure object is the correct class
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[13]:


#this example probability around 1/10, so -ln(1.10) is about 2.3
loss_fn(y_train[:1],predictions).numpy()


# In[16]:


model.compile(optimizer='adam',loss=loss_fn, metrics=['accuracy'])


# In[17]:


model.fit(x_train, y_train, epochs=5)


# In[19]:


#checks model performance using a validation set
#this example about 98% accurate
model.evaluate(x_test, y_test, verbose=2)


# In[21]:


probability_model = tf.keras.Sequential([
    model, tf.keras.layers.Softmax()
])
probability_model(x_test[:5])


# In[ ]:




