import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
import time
import csv

REPEATER_TRAINING_MAP = {'0':"HADAMARD",'1':"CNOT"}
GHZ_TRAINING_MAP = {'0':'HADAMARD', '1':'CNOT'}
TELEPORT_TRAINING_MAP = {"0":"HADAMARD", '1':"CNOT", '2':"CZ"}
W_STATE_TRAINING_MAP = {'0':"RY", '1':"X", '2':"CNOT"}
ADDER_TRAINING_MAP = {'0':"TOFFOLI", '1':"CNOT"}
maps = [TELEPORT_TRAINING_MAP, W_STATE_TRAINING_MAP, GHZ_TRAINING_MAP, REPEATER_TRAINING_MAP, ADDER_TRAINING_MAP]
circuit_types = ['teleport', 'wstate', 'ghz', 'repeater', 'adder']


def handleDataset(filename, split, trainingSet=[], testSet=[]):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data.append(list(lines))
    csvfile.close()
    dataset = data[0]
    for x in range(len(dataset) - 1):
        if random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
    return


def euclideanDistance(ins1, ins2, length):
    # Finds the euclidean distance bewteen two vectors of length 'length;
    dis = 0
    for x in range(length):
        dis += pow((ins1[x] - ins2[x]), 2)
    return math.sqrt(dis)


def getKNeighbors(trainingSet, test, k):
    # finds the k points nearest a test point
    distances = []
    length = len(test) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(test,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    print("_____")
    print("vector:", test)
    for x in neighbors:
        print(x)
    print("_____")
    return neighbors


def getResponse(neighbors):
    # determines the classes of a vector of neighbors and returns a prediction of a test point's class
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    # Finds the accuracy of a test set prediction
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
 

def KNN(path, split, k, unknown_path, circuit_type):
    # Performs KNN classification given a dataset, a training-test split, and k

    map = maps[circuit_types.index(circuit_type)]
    classes = [map[x] for x in map.keys()]
    prediction_count = [0]*len(classes)

    testingset, trainingset = [], []
    handleDataset(path,split,trainingset,testingset)
    predictions=[]
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k) 
        result = getResponse(neighbors) 
        predictions.append(result)

    accuracy = getAccuracy(testingset, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    trainingset = trainingset + testingset

    # Make predictions on IBM data
    temp_unknown_a, temp_unknown_b = [], []
    handleDataset(unknown_path, 0.5, temp_unknown_a, temp_unknown_b)
    unknown = temp_unknown_a + temp_unknown_b
    for x in range(len(unknown)):
        neighbors = getKNeighbors(trainingset, unknown[x], k)
        result = str(int(getResponse(neighbors)))
        gate_type = map[result]
        prediction_count[classes.index(gate_type)] += 1

    print('Predicting Error Classes..')
    for x in range(len(classes)):
        print(classes[x], ': ', prediction_count[x] / len(unknown))

KNN('sim_data/ghz_300probabilities_.csv', 0.8, 5, 'ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv', 'ghz')
# files from 7.24 after edits
"""
Check baseline accuracies

adder_300fidelities_.csv: 96.16438356164385%
ghz_300fidelities_.csv:  100.0% 
repeater_600fidelities_.csv: 74.39148073022312%
teleport_300fidelities_.csv: 88.91625615763546%
wstate_300fidelities_.csv: 76.91154422788605%

adder_300probabilities_.csv: 98.35616438356163%
ghz_300probabilities_.csv:  100.0% 
repeater_300probabilities_.csv:  97.27463312368972%
teleport_300probabilities_.csv:  80.54919908466819%
wstate_300probabilities_.csv:  97.87234042553192%

# Make predictions on melbourne gates

GHZ Probabilities: 
HADAMARD :  0.8686868686868687
CNOT :  0.13131313131313133

GHZ Fidelities: 
HADAMARD :  0.0
CNOT :  1.0

Adder Probabilities: 
TOFFOLI :  0.020202020202020204
CNOT :  0.9797979797979798

Adder Fidelities: 
TOFFOLI :  0.0
CNOT :  1.0

Teleport Probabilities: 
HADAMARD :  0.42424242424242425
CNOT :  0.5353535353535354
CZ :  0.04040404040404041

Teleport Fidelities: 
HADAMARD :  0.20202020202020202
CNOT :  0.797979797979798
CZ :  0.0

WState Probabilities: 
RY :  0.3838383838383838
X :  0.23232323232323232
CNOT :  0.3838383838383838

WState Fidelities: 
RY :  0.08080808080808081
X :  0.0707070707070707
CNOT :  0.8484848484848485

Repeater Probabilities: 
HADAMARD :  0.0
CNOT :  1.0

Repeater Fidelities: 
HADAMARD :  0.6331658291457286
CNOT :  0.36683417085427134


"""

