#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Training with Tensorflow
#This code reads training data for quantum circuits to generate 
    #a neural network that predicts a gate type

#Roy Pace
#Last Update: 6.17.2020


# In[2]:


#Sites used:

#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
#https://medium.com/@a.ydobon/tensorflow-2-0-load-csv-to-tensorflow-2634f7089651


# In[3]:


import tensorflow as tf
import os
import csv


# In[4]:


#At some point I will go back and try to fix this code to where strings work
#but right now I'm just trying to get it compiled, so I changed the data file:
    #Hadamard=0
    #Hadamard2=1
    #CNOT=2
    #CNOT2=3
    #CNOT3=4
    #ControlZ=5


# In[5]:


#File locations of training and testing data
train_data_loc = "/home/roy/Qiskit/2TeleportTrainingData20HadTrue.csv"
test_data_loc = "/home/roy/Qiskit/2TeleportTrainingData20newTrue.csv"


train_data = "file://{}".format(train_data_loc)
train_data_fp = tf.keras.utils.get_file(fname= os.path.basename(train_data),origin=train_data)


test_data = "file://{}".format(test_data_loc)
test_data_fp = tf.keras.utils.get_file(fname= os.path.basename(test_data),origin=test_data)


# In[6]:


#Finds number of rows of train_data

rows=0
with open(train_data_loc, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        rows+=1


# In[7]:


#Format train_data for the neural network

column_names = ["Value_1", "Value_2", "Gate"]
feature_names = column_names[:-1]
label_name = column_names[-1]

#class_names = ["Hadamard", "Hadamard2", "CNOT", "CNOT2", "CNOT3", "ControlZ"]
class_names = [0,1,2,3,4,5]
classnum = len(class_names)

batch_size = rows
train_dataset = tf.data.experimental.make_csv_dataset(train_data_fp,
                                                     batch_size,
                                                      column_names=column_names,
                                                      label_name=label_name,
                                                      num_epochs=1)

#'Packs' features into an array
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()),axis=1)
    return features, labels

#packs features of each pair into the dataset
train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))


# In[8]:


#Creates model for neural network

#I have no idea what the "optimal" number of layers is, I just
    #messed around and 4 layers seemed as good as any other

model = tf.keras.Sequential([tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation=tf.nn.relu, 
                        input_shape=(35,2)),# input shape required 
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.relu),                          
                                                     
  tf.keras.layers.Dense(classnum)
])


# In[9]:


#Define loss

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)
#print("Loss test: {}".format(l))


# In[10]:


#Define Gradient

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
    file_path, batch_size = 5, label_name = "Predicted", na_value="?",
    num_epochs=1, ignore_errors=True, **kwargs)
    return dataset

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# In[11]:


#Optimization

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

loss_value, grads = grad(model, features, labels)

#print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                         #loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))

#print("Step: {},      Loss: {}".format(optimizer.iterations.numpy(),
                                      #loss(model, features, labels, training=True).numpy()))


# In[12]:


#Training!

train_loss_results = []
train_accuracy_results = []

num_epochs = 251

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss_avg.update_state(loss_value)
        
        epoch_accuracy.update_state(y, model(x, training=True))
        
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    #optional, only if you want to know accuracy before test
    if epoch % 50 ==0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, 
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()
                                                                   ))


# In[13]:


#Graph to visualize changes in accuracy

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


# In[14]:


#Check model with test_data

test_dataset = tf.data.experimental.make_csv_dataset(
    test_data_fp,
    batch_size,
    column_names=column_names,
    label_name='Gate',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

