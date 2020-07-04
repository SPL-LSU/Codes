# Current results @ bottom of file
from random import shuffle, random
#import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
import time
import numpy as np
import os

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


REPEATER_TRAINING_MAP = {'0':'Hadamardcreate','1':'CNOTcreate','2':'CNOT2','3':'Hadamard2',
                            '4':'CNOTEntanglement','5':'HadamardEntanglement', '6':'CNOT3',
                            '7':'Hadamard3', '8':'CNOT4', '9':'Hadamard4', '10':'CNOT5', '11':'Hadamard5',
                            '12':'CNOT6', '13':'Hadamard6', '14':'CNOT7', '15':'Hadamard7', '16':'CNOT8',
                            '17':'Hadamard8','18':'CNOT9'}

GHZ_TRAINING_MAP = {'0':'Hadamard', '1':'CNOT','2':'CNOT2','3':'CNOT3'}

TELEPORT_TRAINING_MAP = {'0':'Hadamard', '1':'CNOT', '2':'CNOT2', '3':'Hadamard2', '4':'CNOT3', '5':'Control Z', '6': 'Ideal'}

W_STATE_TRAINING_MAP = {'0': 'Rotation', '1': 'X', '2': 'X2', '3': 'CNOT', '4': 'Rotation2',
                        '5': 'CNOT2', '6': 'Rotation3', '7': 'X3', '8': 'X4', '9':'CNOT3', '10':'CNOT4'}


column_names = ["Value_1", "Value_2", "Value3", "Value4", "Gate"]

"""
column_names = []
dx = 0
for x in range(31):
    column_names.append(("Value_" + str(dx + 1)))
    dx += 1
column_names.append('Gate')
print(column_names)
"""

feature_names = column_names[:-1]
label_name = column_names[-1]

#===============================================================================#

def get_length(path):
    length = 0
    with open(path,'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data = list(lines)
        length += len(data)
    csvfile.close()
    return length

#===============================================================================#

def make_dataset(train_path, rows):

    dataset = tf.data.experimental.make_csv_dataset(train_path, rows,
                                                          column_names=column_names,
                                                          label_name=label_name,
                                                          num_epochs=1,
                                                          shuffle = True)


    return dataset
#
#===============================================================================#
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
#===============================================================================#

def make_model(train_dataset, rows, classnum,choice):

    def pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))
    act = tf.nn.swish

    if choice in ['Repeater', 'GHZ']:
        model = tf.keras.Sequential([tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation=tf.nn.swish, input_shape=(rows, 6), kernel_regularizer = tf.keras.regularizers.l2(l=0.9)), ##
        tf.keras.layers.Dense(150, activation=tf.nn.swish, kernel_regularizer = tf.keras.regularizers.l2(l=0.9)),
        tf.keras.layers.Dense(75, activation=tf.nn.swish, kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
        tf.keras.layers.Dense(classnum, kernel_regularizer = tf.keras.regularizers.l2(l=0.9))])
        return model, features, labels, train_dataset

    elif choice == 'Teleportation':
        kreg = tf.keras.regularizers.l2(l=0.85)  # 0.08
        act = tf.nn.swish
        model = tf.keras.Sequential()
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
            gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
            beta_constraint=None, gamma_constraint=None, trainable=True, name=None
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 4)))
        tf.keras.layers.Dropout(.05)
        model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 4)))
        tf.keras.layers.Dropout(.05)
        model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 4)))
        tf.keras.layers.Dropout(.05)


    return model, features, labels, train_dataset

#===============================================================================#

def loss(model, x, y):
    y = y.numpy()
    y = np.int_(y)
    tf.convert_to_tensor(y)

    y_ = model(x)
    loss_object = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_,
    )


    return loss_object

#===============================================================================#

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)


    grads = tape.gradient(loss_value, model.trainable_variables)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    return loss_value, grads

#===============================================================================#

def optimize(learning_rate, loss_value, model, grads, features, labels, choice):

    if choice == 'Teleportation':
         #optimizer = tf.keras.optimizers.Nadam(learning_rate=.09, epsilon=1e-12, beta_1=0.999, beta_2=0.99999)
         optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.0, epsilon=1e-10, use_locking=False,centered=True, name='RMSProp')
         #optimizer = tf.compat.v1.train.MomentumOptimizer(
         #   0.6, 0.9, use_locking=False, name='Momentum', use_nesterov=True
         #)

         #optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=1, nesterov=True)

    elif choice in ['GHZ', 'Repeater']:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov = True, momentum = .9)


    return optimizer


#===============================================================================#
def train_model(train_dataset,model,optimizer):
    train_loss_results, train_accuracy_results = [], []

    num_epochs = 600
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

        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()
                                                                        ))

    return train_accuracy_results, train_loss_results

#===============================================================================#
def plotter(train_accuracy_results,train_loss_results):

    fig, axes = plt.subplots(2, sharex = True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()
    return
#===============================================================================#
def test_run(test_dataset, model):
    test_dataset = test_dataset.map(pack_features_vector)
    predictions = []
    targets = []
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

        predictions.append(prediction.numpy().tolist())
        targets.append(y.numpy().tolist())

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


    return predictions, targets
#===============================================================================#

def print_results(predictions, targets, mapper):
    print(type(predictions))
    print(type(targets))
    for x in range(len(predictions)):
        for y in range(len(predictions[x])):
            print("_________")
            #print("Predicted: ", mapper[str(int(predictions[x][y]))])
            print("Predicted: ", predictions[x][y])
            print("Actual: ", targets[x][y])
            #print("Actual: ", mapper[str(int(targets[x][y]))])
            print("_________")

    return

#===============================================================================#

def main(path, mapper, choice):
    acc_results, loss_results = [],[]

    data_loc = path
    test_path, train_path = data_loc[:-4] + "_train.csv", data_loc[:-4] + "_test.csv"
    rows = get_length(train_path)
    batchsize = 60 #len(mapper.keys())


    train_dataset = make_dataset(train_path, rows)
    test_dataset = make_dataset(test_path, rows)

    model, features, labels, train_dataset = make_model(train_dataset, rows, batchsize, choice)
    loss_value, grads = grad(model, features, labels)

    for x in range(1):
        learning_rate = .8 #.09 for repeater
        optimizer = optimize(learning_rate, loss_value, model, grads, features, labels, choice)
        train_accuracy_results, train_loss_results = train_model(train_dataset, model, optimizer)
        acc_results= acc_results + train_accuracy_results
        loss_results= loss_results + train_loss_results

    plotter(acc_results, loss_results)

    predictions, targets = test_run(test_dataset, model)
    #print_results(predictions,targets,mapper)

    return


path = "training_data_with_numerals/July3TeleportTrainingData1000Had.csv"
mapper = TELEPORT_TRAINING_MAP

main(path, mapper,'Teleportation')


"""new
[0.7437184400574876, 0.7437184400574876, 0.7437184400574876, 0.5625000046098119, 'Ideal']
QFT
[0.562500001972077, 0.43435921991530446, 0.562500001972077, 0.43435922326859466, 'Ideal']
Hadamard 2
[0.562500001972077, 0.43435921816358347, 0.562500001972077, 0.5291972992189105, 'Ideal']
Fourier State
[0.562500001972077, 0.43435921991530446, 0.562500001972077, 0.43435922326859466, 'Ideal']
23.866217851638794
"""

"""

Reqires datafiles with numeric classes but fix_data_files.py will build these
And split_data_files.py splits into train/test files

# took out frequency balancing and robust scaling
Results atm: (Altering only optimizer parameters and batchsize)

July2TeleportTrainingData1600Had.csv: Train: 85.470% Test: 81.780% (batchsize = 60, 600 epochs)
     copy-paste: optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.009, momentum=0.001, epsilon=1e-10, use_locking=True,centered=True, name='RMSProp')

July2TeleportTrainingData1600new.csv: Train: 52.925%  Test: 51.367% (batchsize = 60, 600 epochs)
     copy-paste: optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.09, momentum=0.009, epsilon=1e-10, use_locking=True,centered=True, name='RMSProp')

July2TeleportTrainingData1600QFT.csvL Train: 73.342%  Test: 69.466% (batchsize = 60, 600 epochs)
     copy-paste: optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.009, momentum=0.005, epsilon=1e-10, use_locking=True,centered=True, name='RMSProp')

July2TeleportTrainingData1600QFT2.csv Train: 73.316% Test:  67.448% (batchsize = 60, 600 epochs)
     copu-paste: optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False,centered=True, name='RMSProp') 


RepeaterTrainingDataQFT2_June_21_num.csv
with learn rate = .9, all l2 regularizers = .9, 10000 epochs,
layers (250, 150, 75):

Epoch 9900: Loss: 0.585, Accuracy: 74.455%
Test set accuracy: 72.647%

GHZTrainingDataHad2_June21_num.csv
with same settings

Epoch 700: Loss: 0.078, Accuracy: 96.825%
Test set accuracy: 64.894%

Jun22OneQubitAdder_600.csv
Same settings with map/column names changed
Epoch 400: Loss: 0.153, Accuracy: 89.189%
Test set accuracy: 92.105%
(but loses 8 classes out of 47 even with all basis states as features )

June24TeleportTrainingData400QFT2

"""
