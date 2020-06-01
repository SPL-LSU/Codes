#!/usr/bin/env python
# coding: utf-8

# In[8]:


#https://www.tensorflow.org/tutorials/keras/classification


# In[9]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[10]:


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[12]:


#import fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#loads a set of 28x28 images with labels ranging from 0 to 9
#need to label labels
class_names = ["T-Shirt", "Pants", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# In[16]:


train_images.shape


# In[15]:


len(train_labels)


# In[17]:


train_labels


# In[18]:


test_images.shape


# In[19]:


len(test_labels)


# In[23]:


#need to preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
#scale pixels to a range of 0 to 1 before processing data


# In[24]:


train_images = train_images/255.
test_images = test_images/255.


# In[32]:


plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,1+i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[35]:


#makes 3 layer network
#first layer turns 2d array (28x28) into 1d array (784)
#2nd and 3rd layer densely connected
#3rd returns logits array with length of 10

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10)
])

#sets up model to use Adam optimizer and prioritize accuracy
model.compile(optimizer='adam', 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])


# In[36]:


#feed model
model.fit(train_images, train_labels, epochs=10)


# In[37]:


#evaluate accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy', test_acc)


# In[38]:


probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()])


# In[39]:


predictions = probability_model.predict(test_images)


# In[40]:


predictions[0]


# In[41]:


np.argmax(predictions[0])


# In[43]:


#to graph predictions

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel("{}{:2.0f}% ({})".format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[51]:


for i in range(25):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


# In[ ]:




