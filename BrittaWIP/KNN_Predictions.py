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

REPEATER_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"HADAMARD2", '2':"CNOT1", '3':"CNOT2", '4':"HADAMARD3", '5':"HADAMARD4", '6':"HADAMARD5",
                             '7':"HADAMARD6", '8':"CNOT3", '9':"HADAMARD7", '10':"HADAMARD8", '11':"HADAMARD9", '12':"CNOT4", '13':"HADAMARD10",
                             '14':"HADAMARD11", '15':"CNOT5", '16':"HADAMARD12", '17':"HADAMARD13", '18':"CNOT6", '19':"HADAMARD14", '20':"HADAMARD15",
                             '21':"CNOT7", '22':"HADAMARD16", '23':"CNOT8", '24':"HADAMARD17", '25':"HADAMARD18", '26':"CNOT9", '27':"HADAMARD19",
                             '28':"HADAMARD20", '29':"CNOT10", '30':"HADAMARD21", '31':"HADAMARD22", '32':"CNOT11"}
GHZ_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"CNOT1", '2':"CNOT2", '3':"CNOT3"}
TELEPORT_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"HADAMARD2", '2':"CNOT1", '3':"CNOT2", '4':"HADAMARD3", '5':"CNOT3", '6':"CZ1"}
W_STATE_TRAINING_MAP_ALL = {'0':"RY1", '1':"X1", '2':"X2", '3':"CNOT1", '4':"RY2", '5':"CNOT2", '6':"RY3", '7':"X3", '8':"X4", '9':"CNOT3", '10':"CNOT4"}
ADDER_TRAINING_MAP_ALL = {'0':"TOFFOLI1", '1':"TOFFOLI2", '2':"TOFFOLI3", '3':"CNOT1", '4':"CNOT2", '5':"CNOT3"}

maps_all = [TELEPORT_TRAINING_MAP_ALL, W_STATE_TRAINING_MAP_ALL, GHZ_TRAINING_MAP_ALL, REPEATER_TRAINING_MAP_ALL, ADDER_TRAINING_MAP_ALL]


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
        dist = euclideanDistance(test[:-1],trainingSet[x][:-1],length) # changed indexing to slice out the class
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    #print("_____")
    #print("vector:", test)
    #for x in neighbors:
    #    print(x)
    #print("_____")
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

    map = maps_all[circuit_types.index(circuit_type)] # maps has to be changed to 'maps_all' for the index-based learning
    print(map)
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

KNN('sim_data/repeater_800fidelities_allgates_.csv', 0.9, 15, 'ibm_data/repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv', 'repeater')
# files from 7.24 after classical sim edits for class-based
# files from 7.25 for index based
"""
Check baseline accuracies on class-based data # will have to run these again, made tiny modifications but probably won't change anything
# All these were k = 5 and split = 0.8 
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
"""
changed split to 0.9 for these 
check baseline accuracies on index-based data 
teleport_800probabilities_allgates_.csv: 56.305506216696266% k = 20 
teleport_1000fidelities_allgates_.csv: 85.96491228070175%    k = 15 
repeater_800probabilities_allgates_.csv' 50.804597701149426% k = 15 
wstate_800fidelities_allgates_.csv  68.67196367763904%       k = 15
wstate_800probabilities_allgates_.csv  97.54959159859978%    k = 15
ghz_800fidelities_allgates_.csv  99.70760233918129%          k = 15
ghz_800probabilities_allgates_.csv 99.70930232558139%        k = 15
adder_800fidelities_allgates_.csv 91.70212765957447%         k = 15 
adder_800probabilities_allgates_.csv 98.53249475890985%      k = 15 
repeater_800fidelities_allgates_.csv 32.86454478164323%      k = 15  #could try gathering more data here

# Make predictions on Melbourne Gates 

Teleport Fidelities 
Predicting Error Classes.. k = 20 
HADAMARD1 :  0.1111111111111111
HADAMARD2 :  0.0
CNOT1 :  0.16161616161616163
CNOT2 :  0.24242424242424243
HADAMARD3 :  0.13131313131313133
CNOT3 :  0.35353535353535354
CZ1 :  0.0


Teleport Probabilities
Predicting Error Classes.. k = 15
HADAMARD1 :  0.030303030303030304 
HADAMARD2 :  0.030303030303030304
CNOT1 :  0.25252525252525254
CNOT2 :  0.2222222222222222
HADAMARD3 :  0.40404040404040403
CNOT3 :  0.030303030303030304
CZ1 :  0.030303030303030304


W-State Fidelities  k = 15
Predicting Error Classes..
RY1 :  0.18181818181818182
X1 :  0.0
X2 :  0.010101010101010102
CNOT1 :  0.0
RY2 :  0.0
CNOT2 :  0.08080808080808081
RY3 :  0.0
X3 :  0.0
X4 :  0.010101010101010102
CNOT3 :  0.494949494949495
CNOT4 :  0.2222222222222222


W-State Probabilities k = 15 
Predicting Error Classes..
RY1 :  0.0
X1 :  0.010101010101010102
X2 :  0.0
CNOT1 :  0.0
RY2 :  0.42424242424242425
CNOT2 :  0.0
RY3 :  0.0
X3 :  0.23232323232323232
X4 :  0.0
CNOT3 :  0.0
CNOT4 :  0.3333333333333333


GHZ Fidelities 
HADAMARD1 :  0.0
CNOT1 :  0.15151515151515152
CNOT2 :  0.13131313131313133
CNOT3 :  0.7171717171717171


GHZ Probabilities 
Predicting Error Classes..
HADAMARD1 :  0.9292929292929293
CNOT1 :  0.0
CNOT2 :  0.050505050505050504
CNOT3 :  0.020202020202020204

Adder Fidelities 
TOFFOLI1 :  0.0
TOFFOLI2 :  0.0
TOFFOLI3 :  0.0
CNOT1 :  0.0
CNOT2 :  1.0
CNOT3 :  0.0

Adder Probabilities
TOFFOLI1 :  0.0
TOFFOLI2 :  0.0
TOFFOLI3 :  0.0
CNOT1 :  0.6868686868686869
CNOT2 :  0.24242424242424243
CNOT3 :  0.0707070707070707



# does the error actually happen at this index in the quantum computer, or does it only look this way numerically? 
# does circuit design alter how error manifests (outside of error correcting codes)?
# how much do errors depend on circuits vs. hardware? 
# how much can the discrepancy b/w probabilities and fidelities be exploited for error mitigation? is it practical? 
Repeater Probabilities  # this one is weird, accuracy is low but it specifically predicts only a few 
Predicting Error Classes..
HADAMARD1 :  0.0
HADAMARD2 :  0.0
CNOT1 :  0.9292929292929293
CNOT2 :  0.030303030303030304
HADAMARD3 :  0.0
HADAMARD4 :  0.0
HADAMARD5 :  0.0
HADAMARD6 :  0.0
CNOT3 :  0.0
HADAMARD7 :  0.0
HADAMARD8 :  0.0
HADAMARD9 :  0.0
CNOT4 :  0.0
HADAMARD10 :  0.0
HADAMARD11 :  0.0
CNOT5 :  0.0
HADAMARD12 :  0.0
HADAMARD13 :  0.0
CNOT6 :  0.04040404040404041
HADAMARD14 :  0.0
HADAMARD15 :  0.0
CNOT7 :  0.0
HADAMARD16 :  0.0
CNOT8 :  0.0
HADAMARD17 :  0.0
HADAMARD18 :  0.0
CNOT9 :  0.0
HADAMARD19 :  0.0
HADAMARD20 :  0.0
CNOT10 :  0.0
HADAMARD21 :  0.0
HADAMARD22 :  0.0
CNOT11 :  0.0


Repeater Fidelities
Predicting Error Classes..
HADAMARD1 :  0.0
HADAMARD2 :  0.0
CNOT1 :  0.0
CNOT2 :  0.0
HADAMARD3 :  0.0
HADAMARD4 :  0.0
HADAMARD5 :  0.18592964824120603
HADAMARD6 :  0.0
CNOT3 :  0.0
HADAMARD7 :  0.0
HADAMARD8 :  0.0
HADAMARD9 :  0.0
CNOT4 :  0.271356783919598
HADAMARD10 :  0.10552763819095477
HADAMARD11 :  0.1507537688442211
CNOT5 :  0.01507537688442211
HADAMARD12 :  0.0
HADAMARD13 :  0.0
CNOT6 :  0.0
HADAMARD14 :  0.1507537688442211
HADAMARD15 :  0.0
CNOT7 :  0.0
HADAMARD16 :  0.0
CNOT8 :  0.010050251256281407
HADAMARD17 :  0.0
HADAMARD18 :  0.0
CNOT9 :  0.005025125628140704
HADAMARD19 :  0.0
HADAMARD20 :  0.0
CNOT10 :  0.0
HADAMARD21 :  0.0
HADAMARD22 :  0.0
CNOT11 :  0.10552763819095477

Process finished with exit code 0


"""

