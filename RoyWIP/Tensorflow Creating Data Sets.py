#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
#https://medium.com/@a.ydobon/tensorflow-2-0-load-csv-to-tensorflow-2634f7089651
#Data Set has first line, a header with info about dataset
#rest of data has features in first columns, label in last


# In[2]:


import tensorflow as tf
import os


# In[3]:


train_data = "/home/roy/Qiskit/2TeleportTrainingData20Had.csv"
train_data_fp = tf.keras.utils.get_file(fname= os.path.basename(train_data),origin=train_data)
print("Copy: {}".format(train_data_fp))


# In[ ]:




