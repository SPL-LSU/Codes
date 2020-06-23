"""
Adaptation of Roy's Tensorflow_Training code.
Reqires datafiles with numeric classes but fix_data_files.py will build these
And split_data_files.py splits into train/test files


Results atm:

RepeaterTrainingDataQFT2_June_21_num.csv
with learn rate = .9, all l2 regularizers = .9, 10000 epochs,
layers (250, 150, 75):

Epoch 9900: Loss: 0.585, Accuracy: 74.455%
Test set accuracy: 72.647%


GHZTrainingDataHad2_June21_num.csv
with same settings

Epoch 700: Loss: 0.078, Accuracy: 96.825%
Test set accuracy: 64.894%


"""





from random import shuffle, random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
import time
import numpy as np



REPEATER_TRAINING_MAP = {'0':'Hadamardcreate','1':'CNOTcreate','2':'CNOT2','3':'Hadamard2',
                            '4':'CNOTEntanglement','5':'HadamardEntanglement', '6':'CNOT3',
                            '7':'Hadamard3', '8':'CNOT4', '9':'Hadamard4', '10':'CNOT5', '11':'Hadamard5',
                            '12':'CNOT6', '13':'Hadamard6', '14':'CNOT7', '15':'Hadamard7', '16':'CNOT8',
                            '17':'Hadamard8','18':'CNOT9'}

GHZ_TRAINING_MAP = {'0':'Hadamard', '1':'CNOT','2':'CNOT2','3':'CNOT3'}

TELEPORT_TRAINING_MAP = {'0':'Hadamard', '1':'CNOT', '2':'CNOT2', '3':'Hadamard2', '4':'CNOT3', '5':'Control Z'}

column_names = ["Value_1", "Value_2", "Value3", "Value4", "Gate"]
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

    dataset = tf.data.experimental.make_csv_dataset(train_path,
                                                          rows,
                                                          column_names=column_names,
                                                          label_name=label_name,
                                                          num_epochs=1,
                                                          shuffle = True)

    return dataset

#===============================================================================#
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
#===============================================================================#

def make_model(train_dataset, rows, classnum):
    def pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))

    model = tf.keras.Sequential([tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation=tf.nn.swish, input_shape=(rows, 4), kernel_regularizer = tf.keras.regularizers.l2(l=0.9)),
    tf.keras.layers.Dense(150, activation=tf.nn.swish, kernel_regularizer = tf.keras.regularizers.l2(l=0.9)),
    tf.keras.layers.Dense(75, activation=tf.nn.swish, kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
    tf.keras.layers.Dense(classnum, kernel_regularizer = tf.keras.regularizers.l2(l=0.9))])

    return model, features, labels, train_dataset

#===============================================================================#
def loss(model, x, y, training):
    y_ = model(x)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    return loss_object(y_true=y, y_pred=y_)

def define_loss(model, features, labels):

    l = loss(model, features, labels, training=False)
    print("Loss test: {}".format(l))

    return l

#===============================================================================#

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#===============================================================================#

def optimize(learning_rate, loss_value, model, grads, features, labels):

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov = True, momentum = .9)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                           loss(model, features, labels, training = True).numpy()))

    return optimizer


#===============================================================================#
def train_model(train_dataset,model,optimizer):
    train_loss_results, train_accuracy_results = [], []

    num_epochs = 10000

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

        # optional, only if you want to know accuracy before test
        if epoch % 100 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()
                                                                        ))
    return train_accuracy_results, train_loss_results

#===============================================================================#
def plotter(train_accuracy_results,train_loss_results):
    # Graph to visualize changes in accuracy
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
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
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
            print("Predicted: ", mapper[str(int(predictions[x][y]))])
            print("Actual: ", mapper[str(int(targets[x][y]))])
            print("_________")

    return

#===============================================================================#

def main(path, mapper):
    acc_results, loss_results = [],[]

    data_loc = path
    test_path, train_path = data_loc[:-4] + "_train.csv", data_loc[:-4] + "_test.csv"
    rows = get_length(train_path)
    classnum = len(mapper.keys())

    train_dataset = make_dataset(train_path, rows)
    test_dataset = make_dataset(test_path, rows)

    model, features, labels, train_dataset = make_model(train_dataset, rows, classnum)
    l = define_loss(model, features, labels)
    loss_value, grads = grad(model, features, labels)

    for x in range(1):
        learning_rate = .09
        optimizer = optimize(learning_rate, loss_value, model, grads, features, labels)
        train_accuracy_results, train_loss_results = train_model(train_dataset, model, optimizer)
        acc_results= acc_results + train_accuracy_results
        loss_results= loss_results + train_loss_results

        print('Going to sleep...')
        time.sleep(5)

    plotter(acc_results, loss_results)

    predictions, targets = test_run(test_dataset, model)
    print_results(predictions,targets,mapper)

    return


path = "training_data_with_numerals/June21/GHZTrainingDataHad2_June21_num.csv"
mapper = REPEATER_TRAINING_MAP

main(path, mapper)