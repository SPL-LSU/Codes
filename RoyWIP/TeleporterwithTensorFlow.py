#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow_datasets.public_api as tfds


# In[ ]:


#https://www.tensorflow.org/datasets/add_dataset
#https://danijar.com/structuring-models/
#https://danijar.com/structuring-your-tensorflow-models/
#https://www.bmc.com/blogs/keras-neural-network-classification/


# In[2]:


class MyDataset(tfds.core.GeneratorBasedBuilder):
    #Quantum Teleportation Dataset
    
    VERSION = tfds.core.Version('0.1.0')
    
    def _info(self):
        builder = self,
        description=("Dataset for the teleportation code. It contains Qobj")
        #specifies the tfds.core.DatasetInfo object
        #tfds.features.FeatureConnectors
        features = tfds.features.FeaturesDict({
            "object description": tfds.features.Text(),
            "object": tfds.features.Image(),
            "label": tfds.features.ClassLabel(num_classes = 6),
        }),
        supervised_keys=('image', 'label'),
        
    
    def _split_generators(self, dl_manager):
        #Downloads data and defines split
        #dl_manager is a tfds.download.DownloadManager 
             #that can be used to download and extract urls
            
        pass #TODO
    
    def _generate_examples(self):
        #yields examples from the dataset
        yield 'key', {}


# In[6]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
import time
import csv

print(tf.__version__)


# In[7]:


#splits the data set into testing and training data. Need to initialize empty vectors first
def handleDataset(array,split,trainingSet=[],testSet=[]):
    #with open(filename,'r') as csvfile:
        #lines = csv.reader(csvfile)
        #dataset=list(lines)
    dataset=array
    for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y]=float(dataset[x][y])
            if random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        #print(trainingSet, 'aaaahhh', testSet)
    return 0


# In[22]:


with open('2TeleportTrainingData20new.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    trainingdata = []
    for row in csv_reader:
        trainingdata.append(row)
        
with open('2TeleportTrainingData20QFT.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    testdata = []
    for row in csv_reader:
        testdata.append(row)
        
trainingdata=np.array(trainingdata)
testdata=np.array(testdata)


# In[30]:


#traininglabels


# In[29]:


traininglabels = []
for i in range(len(trainingdata)):
    traininglabels.append(trainingdata[i,2])


# In[25]:


model = keras.Sequential([
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(6)
])

model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])


# In[55]:


#model.fit((train0,train1), train_labels, epochs=1 )


# In[37]:


train_labels=[traininglabels]
train_data = [trainingdata]


# In[52]:


train0 = []
train1 = []
for i in range(len(trainingdata)):
    train0.append(trainingdata[i,0])
    train1.append(trainingdata[i,1])


# In[53]:


train0


# In[ ]:




