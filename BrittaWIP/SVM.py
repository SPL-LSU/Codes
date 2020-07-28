# SVM for gate-based classification
# This is to replace the previous model, which had a fatal flaw
# B Manifold 7. 26
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import csv
import random
import numpy as np
from sklearn.metrics import accuracy_score

# The data is not scaled because the sklearn scalers do something messed up to the results.

#===================================================================Broad Gate Classification=============================
# Easier circuits are run together, harder circuits run separately

"""
# would need to be uncommented 
repeater = {'0':"HADAMARD",'1':"CNOT"}
ghz = {'0':'HADAMARD', '1':'CNOT'}
teleport = {"0":"HADAMARD", '1':"CNOT", '2':"CZ"}
w_state = {'0':"RY", '1':"X", '2':"CNOT"}
adder = {'0':"TOFFOLI", '1':"CNOT"}
"""

# Repeater
"""
files = [('sim_data/by_class/repeater_600fidelities_.csv', 'ibm_data/repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/repeater_300probabilities_.csv', 'ibm_data/repeater100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]
         
circ_names = ['Repeater', 'Repeater']
maps = [ repeater, repeater]
metrics = ['fidelities', 'probabilities']
"""
# WState
"""
files = [('sim_data/by_class/wstate_300fidelities_.csv', 'ibm_data/wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/wstate_300probabilities_.csv', 'ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]

circ_names = ['WState', 'WState']
maps = [w_state, w_state]
metrics = ['fidelities', 'probabilities']
"""

# Last-minute add-on for deutsch-jozsa (whole gate classes )

deutsch = {'0':'HADAMARD', '1':'X', '2':'CNOT'}
files = [('sim_data/deutsch_300fidelities_.csv','ibm_data/deutsch100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
        ('sim_data/deutsch_300probabilities_.csv','ibm_data/deutsch100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]
circ_names = ['Deutsch', 'Deutsch'] 
maps = [deutsch, deutsch]
metrics = ['fidelities', 'probabilities']



# The Rest
"""
files = [('sim_data/by_class/ghz_300fidelities_.csv', 'ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/ghz_300probabilities_.csv', 'ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/teleport_300fidelities_.csv', 'ibm_data/teleport100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/teleport_300probabilities_.csv', 'ibm_data/teleport100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/by_class/adder_300fidelities_.csv', 'ibm_data/adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/by_class/adder_300probabilities_.csv', 'ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]

circ_names = ['GHZ', 'GHZ', 'Teleport', 'Teleport', 'Adder', 'Adder']
maps = [ghz, ghz, teleport, teleport, adder, adder]
metrics = ['fidelities', 'probabilities']*3
"""
# ======================================================== Specific Gate Classification =============================================#
"""
ghz = {'0':"HADAMARD1", '1':"CNOT1", '2':"CNOT2", '3':"CNOT3"}
wstate= {'0':"RY1", '1':"X1", '2':"X2", '3':"CNOT1", '4':"RY2", '5':"CNOT2", '6':"RY3", '7':"X3", '8':"X4", '9':"CNOT3", '10':"CNOT4"}
adder = {'0':"TOFFOLI1", '1':"TOFFOLI2", '2':"TOFFOLI3", '3':"CNOT1", '4':"CNOT2", '5':"CNOT3"}

files = [('sim_data/ghz_800fidelities_allgates_.csv', 'ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/ghz_800probabilities_allgates_.csv', 'ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/adder_800fidelities_allgates_.csv', 'ibm_data/adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/adder_800probabilities_allgates_.csv', 'ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv'),
         ('sim_data/wstate_800fidelities_allgates_.csv', 'ibm_data/wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
         ('sim_data/wstate_800probabilities_allgates_.csv', 'ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]

circ_names = ['GHZ', 'GHZ', 'Adder', 'Adder', 'WState', 'WState']
maps = [ghz, ghz, adder, adder, wstate, wstate]
metrics = ['fidelities', 'probabilities']*3
"""
# Last minute add-on for deutsch-jozsa (single gates)
"""
deutsch = {'0':'HADAMARD1', '1':'HADAMARD2', '2':'HADAMARD3', '3':'HADAMARD4', '4':'X1', '5':'HADAMARD5', '6':'X2', '7':'X3', '8':'CNOT1', '9':'X4',
            '10':'CNOT2', '11':'CNOT3', '12':'X5', '13':'CNOT4', '14':'HADAMARD6', '15':'HADAMARD7', '16':'HADAMARD8', '17':'HADAMARD9'}
files = [('sim_data/deutsch_800fidelities_allgates_.csv','ibm_data/deutsch100_ibm_sim_fidelitiesibmq_16_melbourne.csv'),
        ('sim_data/deutsch_800probabilities_allgates_.csv','ibm_data/deutsch100_ibm_sim_probabilities_ibmq_16_melbourne.csv')]
circ_names = ['Deutsch', 'Deutsch']
maps = [deutsch, deutsch]
metrics = ['fidelities', 'probabilities']
"""
# ======================================================== Code======================================================================#
def handleDataset(filename, split, train_set=[], test_set=[]):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data.append(list(lines))
    csvfile.close()
    dataset = data[0]
    for x in range(len(dataset) - 1):
        if random.random() < split:
            train_set.append(dataset[x])
        else:
            test_set.append(dataset[x])


def read_file(filename):
    data=[]
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data.append(list(lines))
    dataset = data[0]
    return dataset


def scale_data(train_set, test_set):

    X_train = np.array(train_set)[:,:-1]
    y_train = (np.array(train_set)[:, -1])
    X_test = np.array(test_set)[:, :-1]
    y_test = (np.array(test_set)[:, -1])
    return X_train, y_train, X_test, y_test


def get_model(X_train, y_train):

    svm = SVC(kernel='poly', degree=3, gamma='auto', C=1000.0, decision_function_shape='ovr', class_weight='balanced', probability=True)
    svm.fit(X_train, y_train)
    return svm


def get_accuracy(svm, X_test, y_test, accuracies, i):

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


def predict(X_ibm, svm, predictions, map):
    preds = svm.predict(X_ibm)
    for x in list(preds):
        predictions.append(map[str(int(x))])
    return


def designate_predictions(predictions_total):
    # Find the overall frequency of class predictions on the unknown data

    print("Predicting Unknowns....")
    count = 0
    frequencies = {}
    for x in predictions_total:
        if x not in frequencies.keys():
            frequencies[x] = 1
            count += 1
        else:
            frequencies[x] += 1
            count += 1
    for x in frequencies.keys():
        print(x, ":", frequencies[x] / count)


def main():

    for x in range(len(files)):
        accuracy_scores = []
        predictions = []

        sim_data, ibm_data = files[x][0], files[x][1]
        circuit = circ_names[x]
        map = maps[x]
        metric = metrics[x]

        print('____________________________________')
        print('Circuit:', circuit, '; Metric:', metric)
        for i in range(2):

            train_set, test_set = [], []
            handleDataset(sim_data, 0.9, train_set, test_set)
            X_train, y_train, X_test, y_test = scale_data(train_set, test_set)

            ibm_dataset = read_file(ibm_data)
            X_ibm = np.array(ibm_dataset)[:,:-1]

            # build model and find most prevalent predictions
            svm = get_model(X_train, y_train)
            get_accuracy(svm, X_test, y_test, accuracy_scores, i)
            predict(X_ibm, svm, predictions, map)

        designate_predictions(predictions)
        print('Average Accuracy: ', sum(accuracy_scores)/2)

main()

"""
with SVC(kernel='poly', degree=3, gamma='auto', C=1000.0, decision_function_shape='ovr', class_weight='balanced', probability=True)_______________________________
Circuit: GHZ ; Metric: fidelities
Predicting Unknowns....
CNOT : 0.96
HADAMARD : 0.04
Average Accuracy:  1.0
____________________________________
Circuit: GHZ ; Metric: probabilities
Predicting Unknowns....
CNOT : 1.0
Average Accuracy:  0.7491475409836066
____________________________________
Circuit: Teleport ; Metric: fidelities
Predicting Unknowns....
CNOT : 0.865
HADAMARD : 0.135
Average Accuracy:  0.8966263621436035
____________________________________
Circuit: Teleport ; Metric: probabilities
Predicting Unknowns....
HADAMARD : 1.0
Average Accuracy:  0.780169120939912
____________________________________
Circuit: Adder ; Metric: fidelities
Predicting Unknowns....
CNOT : 1.0
Average Accuracy:  0.9466562389328297
____________________________________
Circuit: Adder ; Metric: probabilities
Predicting Unknowns....
TOFFOLI : 1.0
Average Accuracy:  0.510255018990776

with SVC(kernel='rbf', degree=3, gamma='auto', C=1000.0, decision_function_shape='ovr', class_weight='balanced', probability=True) 
____________________________________
Circuit: WState ; Metric: fidelities
Predicting Unknowns....
CNOT : 0.91
RY : 0.09
Average Accuracy:  0.7867840285750733
____________________________________
Circuit: WState ; Metric: probabilities
Predicting Unknowns....
X : 0.35
CNOT : 0.535
RY : 0.115
Average Accuracy:  0.9019002925755337

SVC(kernel='rbf', degree=3, gamma='auto', C=1000.0, decision_function_shape='ovr', class_weight='balanced', probability=True)
____________________________________
Circuit: Repeater ; Metric: fidelities
Predicting Unknowns....
CNOT : 0.56
HADAMARD : 0.44
Average Accuracy:  0.6876131800645618
____________________________________
Circuit: Repeater ; Metric: probabilities
Predicting Unknowns....
HADAMARD : 0.69
CNOT : 0.31
Average Accuracy:  0.9462107690830701

SVC(kernel='poly', degree=3, gamma='auto', C=1000.0, decision_function_shape='ovr', class_weight='balanced', probability=True)
____________________________________
Circuit: Deutsch ; Metric: fidelities
Predicting Unknowns....
HADAMARD : 0.645
X : 0.355
Average Accuracy:  0.5886900700245782
____________________________________
Circuit: Deutsch ; Metric: probabilities
Predicting Unknowns....
X : 0.5
HADAMARD : 0.5
Average Accuracy:  0.39278063265223107

"""
"""
# ============ Specific gate errors ======

____________________________________
Circuit: GHZ ; Metric: fidelities
Predicting Unknowns....
CNOT2 : 0.45
CNOT3 : 0.54
CNOT1 : 0.01
Average Accuracy:  0.9933331096121794
____________________________________
Circuit: GHZ ; Metric: probabilities
Predicting Unknowns....
CNOT2 : 0.33
HADAMARD1 : 0.5
CNOT1 : 0.1
CNOT3 : 0.07
Average Accuracy:  0.9954407294832827
____________________________________
Circuit: Adder ; Metric: fidelities
Predicting Unknowns....
CNOT2 : 1.0
Average Accuracy:  0.9451607060528222
____________________________________
Circuit: Adder ; Metric: probabilities
Predicting Unknowns....
CNOT2 : 0.435
TOFFOLI3 : 0.3
CNOT1 : 0.23
CNOT3 : 0.015
TOFFOLI2 : 0.02
Average Accuracy:  0.9751346979244127
____________________________________
Circuit: WState ; Metric: fidelities
Predicting Unknowns....
CNOT3 : 0.55
X2 : 0.04
CNOT4 : 0.265
RY1 : 0.07
CNOT2 : 0.065
X1 : 0.01
Average Accuracy:  0.7552322801338875
____________________________________
Circuit: WState ; Metric: probabilities
Predicting Unknowns....
X3 : 0.33
CNOT4 : 0.545
RY2 : 0.095
CNOT2 : 0.02
X1 : 0.01
Average Accuracy:  0.956807539404833

____________________________________
Circuit: Deutsch ; Metric: fidelities
Predicting Unknowns....
X1 : 0.88
HADAMARD6 : 0.075
HADAMARD8 : 0.04
HADAMARD7 : 0.005
Average Accuracy:  0.378073774415016

____________________________________
Circuit: Deutsch ; Metric: probabilities
Predicting Unknowns....
CNOT1 : 0.5
X2 : 0.5
Average Accuracy:  0.0596547820747109

"""
