# Compare performance of various basic classifiers with regional errors vs. single-gate errors
# 7.8.20 BNManifold
#scikit-learn.org/stable/modules/multiclass.html
"""Region1:  0 (1st 3 gates)
Region2:  1 (last 3 gates)"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# single-gate data
path = 'training_data_with_numerals/TeleportTrainingData500Had_train.csv'
test_path = 'training_data_with_numerals/TeleportTrainingData500Had_test.csv'
# regional data
path2 = 'training_data_with_numerals/TeleportTrainingData500Had_Reg_train.csv'
test_path2 = 'training_data_with_numerals/TeleportTrainingData500Had_Reg_test.csv'
data_raw_train = np.loadtxt(path, delimiter=',')
data_raw_test = np.loadtxt(test_path, delimiter=',')
data_raw_trainML = np.loadtxt(path2, delimiter=',')
data_raw_testML = np.loadtxt(test_path2, delimiter=',')

def split_data_target(data):

    dataset = np.array([x[:-1] for x in data.tolist()])
    targets = np.array([x[-1] for x in data.tolist()])

    return dataset, targets


# single-gate data
dataset_train, targets_train = split_data_target(data_raw_train)
dataset_test, targets_test = split_data_target(data_raw_test)
# regional data
dataset_trainML, targets_trainML = split_data_target(data_raw_trainML)
dataset_testML, targets_testML = split_data_target(data_raw_testML)


# Decision Trees
print('Testing Decision Trees (single-gate)...')
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', criterion='gini', splitter='random')
clf.fit(dataset_train, targets_train)
print(cross_val_score(clf, dataset_test, targets_test, cv=4))
print('Testing Decision Trees (regional)...')
clfb = DecisionTreeClassifier(random_state=0, class_weight='balanced', criterion='gini', splitter='random')
clfb.fit(dataset_trainML, targets_trainML)
print(cross_val_score(clfb, dataset_testML, targets_testML, cv=4))
print('\n')

# KNN
print('Testing KNN (single gate)...')
neigh = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
neigh.fit(dataset_train, targets_train)
print(cross_val_score(neigh, dataset_test, targets_test))
print('Testing KNN (regional)...')
neighb = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
neighb.fit(dataset_trainML, targets_trainML)
print(cross_val_score(neighb, dataset_testML, targets_testML))
print('\n')

# Neural Network
print('Testing Neural Network (single gate)...')
clf2 = MLPClassifier(random_state=1, max_iter=3000, hidden_layer_sizes=(10,10), solver='lbfgs',
                     alpha=.00001, activation='tanh').fit(dataset_train, targets_train)
print(clf2.score(dataset_test, targets_test))
print('Testing Neural Network (regional)')
clf2b = MLPClassifier(random_state=1, max_iter=3000, hidden_layer_sizes=(10,10), solver='lbfgs',
                     alpha=.00001, activation='tanh').fit(dataset_trainML, targets_trainML)
print(clf2b.score(dataset_testML, targets_testML))
print('\n')

# Random Forest
print('Testing Random Forest (gate)...')
clf3 = RandomForestClassifier(max_depth=2, random_state=0)
clf3.fit(dataset_train, targets_train)
print(clf3.score(dataset_test, targets_test))
print('Testing Random Forest (regional)...')
clf3b = RandomForestClassifier(max_depth=2, random_state=0)
clf3b.fit(dataset_trainML, targets_trainML)
print(clf3b.score(dataset_testML, targets_testML))
print('\n')

# Ridge Classifier
print('Testing Ridge Classifier (single gate)...')
from sklearn.linear_model import RidgeClassifierCV
clf4 = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=4).fit(dataset_train, targets_train)
print(clf4.score(dataset_test, targets_test))
print('Testing Ridge Classifier (regional)...')
clf4b = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=4).fit(dataset_trainML, targets_trainML)
print(clf4b.score(dataset_testML, targets_testML))
print('\n')

# One-Versus-Rest SVC
print('Testing One-Versus-Rest SVC (single gate)')
clf5 = OneVsRestClassifier(SVC()).fit(dataset_train, targets_train)
print(clf5.score(dataset_test, targets_test))
print('Testing One-Versus-Rest SVC (regional)')
clf5b = OneVsRestClassifier(SVC()).fit(dataset_trainML, targets_trainML)
print(clf5b.score(dataset_testML, targets_testML))
print('\n')

"""
Testing Decision Trees (single-gate)...
[0.22222222 0.44444444 0.33333333 0.375     ]
Testing Decision Trees (regional)...
[0.70909091 0.89090909 0.87272727 0.92592593]


Testing KNN (single gate)...
[0.         0.57142857 0.57142857 0.28571429 0.42857143]
Testing KNN (regional)...
[0.70454545 0.86363636 0.88636364 0.90909091 0.95348837]


Testing Neural Network (single gate)...
0.42857142857142855
Testing Neural Network (regional)
0.8493150684931506


Testing Random Forest (gate)...
0.3142857142857143
Testing Random Forest (regional)...
0.776255707762557


Testing Ridge Classifier (single gate)...
0.3142857142857143
Testing Ridge Classifier (regional)...
0.7031963470319634


Testing One-Versus-Rest SVC (single gate)
0.45714285714285713
Testing One-Versus-Rest SVC (regional)
0.776255707762557
"""

