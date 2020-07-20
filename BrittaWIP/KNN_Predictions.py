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

KNN('sim_data/teleport_100fidelities_.csv', 0.8, 5, 'ibm_data/teleport100_ibm_sim_fidelitiesibmq_16_melbourne.csv', 'teleport')

"""
Check baseline accuracies

ghz_100probabilities_.csv : 98.87640449438202%
adder_60probabilities_.csv: 95.8904109589041%
repeater_40probabilities_.csv  : 91.76029962546816%
teleport_100probabilities_.csv: 95.45454545454545%
wstate_100probabilities_.csv : 87.13692946058092%

adder_60fidelities_.csv : 88.60759493670885%
ghz_100fidelities_.csv: 100.0%
repeater_40fidelities_.csv: 76.37795275590551%
teleport_100fidelities_.csv: 49.333333333333336%
wstate_100fidelities_.csv: 56.837606837606835%

# Make predictions on melbourne gates
_____
GHZ 
ibm data: ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; sim data: ghz_100probabilities_.csv 
HADAMARD :  0.21212121212121213
CNOT :  0.7878787878787878
ibm data: ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv; sim data: ghz_100fidelities_.csv
HADAMARD :  0.0
CNOT :  1.0
_____
WSTATE
ibm data: wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; sim data: wstate_100probabilities_.csv 
RY :  0.30303030303030304
X :  0.3838383838383838
CNOT :  0.31313131313131315
ibm data: wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; sim data: wstate_100fidelities_.csv
RY :  0.0
X :  0.0
CNOT :  1.0
_____
TELEPORT
ibm data: teleport100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; sim data: teleport_100probabilities_.csv 
HADAMARD :  0.20202020202020202
CNOT :  0.7272727272727273
CZ :  0.0707070707070707
ibm data: teleport100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; sim data: teleport_100fidelities_.csv 
HADAMARD :  0.0
CNOT :  0.030303030303030304
CZ :  0.9696969696969697
_____
ADDER
ibm data: adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; sim data: adder_60probabilities_.csv 
Predicting Error Classes..
TOFFOLI :  0.0
CNOT :  1.0

ibm data: adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; sim data: adder_60fidelities_.csv 
TOFFOLI :  0.5252525252525253
CNOT :  0.47474747474747475
_____
REPEATER
ibm data: repeater100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; sim data: repeater_40probabilities_.csv 
HADAMARD :  0.3838383838383838
CNOT :  0.6161616161616161

ibm data: repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; sim data: repeater_40fidelities_.csv 
HADAMARD :  0.0
CNOT :  1.0
"""