#!/usr/bin/env python
# coding: utf-8

# In[1]:


#A foray into basic ANN


# In[2]:


import numpy as np


# In[6]:


class NeuralNetwork():
    def __init__(self):
        #seed rng
        np.random.seed(1)
        #an array of 3 random numbers fro -1 to 1, the weights
        self.synaptic_weights = 2*np.random.random((3,1))-1
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #derivative to sigmoid function?
        return x*(1-x)
    
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            
            #error rate for back propogation
            error = training_outputs - output
            #finds fidelity between two arrays
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights +=adjustments
            
    def think(self, inputs):
        #passes the inputs to neuron, returns output
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    
if __name__ == "__main__":
        
    #neuron class
    neural_network = NeuralNetwork()
        
    print("Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)
        
    user_input_one = str(input("Input 1: "))
    user_input_two = str(input("Input 2: "))
    user_input_three = str(input("Input 3: "))
        
    print("Considering Data: ", user_input_one, user_input_two, user_input_three)
    print("New Output: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
        


# In[ ]:




