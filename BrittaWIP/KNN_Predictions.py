import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
import time
import csv


# Things needed for whole gate classes
"""
repeater = {'0':"HADAMARD",'1':"CNOT"}
ghz = {'0':'HADAMARD', '1':'CNOT'}
teleport = {"0":"HADAMARD", '1':"CNOT", '2':"CZ"}
w_state = {'0':"RY", '1':"X", '2':"CNOT"}
adder = {'0':"TOFFOLI", '1':"CNOT"}

files = [('sim_data/by_class/ghz_300fidelities_.csv', 'ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/ghz_300probabilities_.csv', 'ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/teleport_300fidelities_.csv', 'ibm_data/teleport100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/teleport_300probabilities_.csv', 'ibm_data/teleport100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/wstate_300fidelities_.csv', 'ibm_data/wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/wstate_300probabilities_.csv', 'ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/adder_300fidelities_.csv', 'ibm_data/adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/adder_300probabilities_.csv', 'ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/repeater_600fidelities_.csv', 'ibm_data/repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/repeater_300probabilities_.csv', 'ibm_data/repeater100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ]

circ_names = ['GHZ', 'GHZ', 'Teleport', 'Teleport', 'WState', 'WState', 'Adder', 'Adder', 'Repeater', 'Repeater']
maps = [ghz, ghz, teleport, teleport, w_state, w_state, adder, adder, repeater, repeater]
metrics = ['fidelities', 'probabilities']*5
"""

# Things needed for single gates
ghz = {'0':"HADAMARD1", '1':"CNOT1", '2':"CNOT2", '3':"CNOT3"}
wstate = {'0':"RY1", '1':"X1", '2':"X2", '3':"CNOT1", '4':"RY2", '5':"CNOT2", '6':"RY3", '7':"X3", '8':"X4", '9':"CNOT3", '10':"CNOT4"}
adder = {'0':"TOFFOLI1", '1':"TOFFOLI2", '2':"TOFFOLI3", '3':"CNOT1", '4':"CNOT2", '5':"CNOT3"}

files = [('sim_data/ghz_800fidelities_allgates_.csv', 'ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/ghz_800probabilities_allgates_.csv', 'ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/wstate_800fidelities_allgates_.csv', 'ibm_data/wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/wstate_800probabilities_allgates_.csv', 'ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/adder_800fidelities_allgates_.csv', 'ibm_data/adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/adder_800probabilities_allgates_.csv', 'ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]

circ_names = ['GHZ', 'GHZ', 'WState', 'WState', 'Adder', 'Adder',]
maps = [ghz, ghz, wstate, wstate, adder, adder]
metrics = ['fidelities', 'probabilities']*3

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
 

def KNN(path, split, k, unknown_path, circuit_type, metric, map):
    # Performs KNN classification given a dataset, a training-test split, and k

    classes = [map[x] for x in map.keys()]
    prediction_count = [0]*len(classes)

    testingset, trainingset = [], []
    handleDataset(path,split,trainingset,testingset)
    predictions=[]
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k) 
        result = getResponse(neighbors) 
        predictions.append(result)

    print('_________________________________________')
    print("Circuit: ", circuit_type, "; Metric:", metric)
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


for x in range(len(files)):
    sim_data, ibm_data = files[x][0], files[x][1]
    circuit = circ_names[x]
    map = maps[x]
    metric = metrics[x]
    KNN(sim_data, 0.9, 15, ibm_data, circuit, metric, map)

# files from 7.24 after classical sim edits for class-based
# files from 7.25 for index based

"""

Broad Gate Data 
_________________________________________
Circuit:  GHZ ; Metric: fidelities # 
Accuracy: 99.09909909909909%
Predicting Error Classes..
HADAMARD :  0.0
CNOT :  1.0
_________________________________________
Circuit:  GHZ ; Metric: probabilities # 
Accuracy: 99.20634920634922%
Predicting Error Classes..
HADAMARD :  0.9797979797979798
CNOT :  0.020202020202020204
_________________________________________
Circuit:  Teleport ; Metric: fidelities
Accuracy: 87.20379146919431%
Predicting Error Classes..
HADAMARD :  0.20202020202020202
CNOT :  0.797979797979798
CZ :  0.0
_________________________________________
Circuit:  Teleport ; Metric: probabilities # 
Accuracy: 88.29268292682927%
Predicting Error Classes..
HADAMARD :  0.5555555555555556
CNOT :  0.4444444444444444
CZ :  0.0
_________________________________________
Circuit:  WState ; Metric: fidelities
Accuracy: 83.01886792452831%
Predicting Error Classes..
RY :  0.18181818181818182
X :  0.0
CNOT :  0.8181818181818182
_________________________________________
Circuit:  WState ; Metric: probabilities #
Accuracy: 94.92537313432837%
Predicting Error Classes..
RY :  0.43434343434343436
X :  0.3434343434343434
CNOT :  0.2222222222222222
_________________________________________
Circuit:  Adder ; Metric: fidelities
Accuracy: 96.19565217391305%
Predicting Error Classes..
TOFFOLI :  0.0
CNOT :  1.0
_________________________________________
Circuit:  Adder ; Metric: probabilities # 
Accuracy: 95.58823529411765%
Predicting Error Classes..
TOFFOLI :  0.0
CNOT :  1.0
_________________________________________
Circuit:  Repeater ; Metric: fidelities
Accuracy: 80.58823529411765%
Predicting Error Classes..
HADAMARD :  0.6432160804020101
CNOT :  0.35678391959798994
_________________________________________
Circuit:  Repeater ; Metric: probabilities # 
Accuracy: 94.74768280123584%
Predicting Error Classes..
HADAMARD :  0.0
CNOT :  1.0
"""


"""
Single-Gate Data 

_________________________________________
Circuit:  GHZ ; Metric: fidelities
Accuracy: 99.66996699669967%
Predicting Error Classes..
HADAMARD1 :  0.0
CNOT1 :  0.15151515151515152
CNOT2 :  0.13131313131313133
CNOT3 :  0.7171717171717171
_________________________________________
Circuit:  GHZ ; Metric: probabilities
Accuracy: 99.68051118210862%
Predicting Error Classes..
HADAMARD1 :  0.9292929292929293
CNOT1 :  0.0
CNOT2 :  0.050505050505050504
CNOT3 :  0.020202020202020204
_________________________________________
Circuit:  WState ; Metric: fidelities
Accuracy: 74.6730083234245%
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
_________________________________________
Circuit:  WState ; Metric: probabilities
Accuracy: 97.7983777520278%
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
_________________________________________
Circuit:  Adder ; Metric: fidelities
Accuracy: 92.66247379454927%
Predicting Error Classes..
TOFFOLI1 :  0.0
TOFFOLI2 :  0.0
TOFFOLI3 :  0.0
CNOT1 :  0.0
CNOT2 :  1.0
CNOT3 :  0.0
_________________________________________
Circuit:  Adder ; Metric: probabilities
Accuracy: 97.25400457665904%
Predicting Error Classes..
TOFFOLI1 :  0.0
TOFFOLI2 :  0.0
TOFFOLI3 :  0.0
CNOT1 :  0.6868686868686869
CNOT2 :  0.24242424242424243
CNOT3 :  0.0707070707070707
"""


