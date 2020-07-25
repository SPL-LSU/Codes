# Neural network testing with IBM data
# Decide unknown classes from performing multiple runs
# 7.18 results at bottom

# Has to write and then delete temp, but won't succeed if stopped early. This is to not need split_data_files.py
# ValueError: Problem inferring types: CSV row has different number of fields than expected. This means it was stopped early, delete files.

# B Manifold 7.14.20

from random import shuffle, random
import matplotlib.pyplot as plt
import os
import csv
import time
import numpy as np
import os
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
tf.enable_eager_execution()


REPEATER_TRAINING_MAP = {'0':"HADAMARD",'1':"CNOT"}
GHZ_TRAINING_MAP = {'0':'HADAMARD', '1':'CNOT'}
TELEPORT_TRAINING_MAP = {"0":"HADAMARD", '1':"CNOT", '2':"CZ"}
W_STATE_TRAINING_MAP = {'0':"RY", '1':"X", '2':"CNOT"}
ADDER_TRAINING_MAP = {'0':"TOFFOLI", '1':"CNOT"}
maps = [TELEPORT_TRAINING_MAP, W_STATE_TRAINING_MAP, GHZ_TRAINING_MAP, REPEATER_TRAINING_MAP, ADDER_TRAINING_MAP]

column_names_ghz_prob = ["V" + str(x + 1) for x in range(16)]
column_names_ghz_fid = ["V" + str(x + 1) for x in range(4)]
column_names_repeater_prob =  ["V" + str(x + 1) for x in range(16)]
column_names_repeater_fid =  ["V" + str(x + 1) for x in range(4)]
column_names_teleport_prob =  ["V" + str(x + 1) for x in range(8)]
column_names_teleport_fid =  ["V" + str(x + 1) for x in range(4)]
column_names_wstate_prob =  ["V" + str(x + 1) for x in range(8)]
column_names_wstate_fid =  ["V" + str(x + 1) for x in range(4)]
column_names_adder_prob = ["V" + str(x + 1) for x in range(32)]
column_names_adder_fid = ["V" + str(x + 1) for x in range(4)]
column_names_list_prob = [column_names_teleport_prob, column_names_wstate_prob, column_names_ghz_prob, column_names_repeater_prob, column_names_adder_prob]
column_names_list_fid = [column_names_teleport_fid, column_names_wstate_fid, column_names_ghz_fid, column_names_repeater_fid, column_names_adder_fid]

for x in column_names_list_prob:
    x.append("Gate")
for x in column_names_list_fid:
    x.append("Gate")
column_names = []
feature_names = []
label_name = []
#===============================================================================#


def get_length(path):

    length = 0
    with open(path,'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data = list(lines)
        length += len(data)
    csvfile.close()
    return length


def write_data(save_path, vectors):

    # temporarily writes data for the input pipeline
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for x in vectors:
            writer.writerow(x)
    csvFile.close()


def make_dataset(path):
    # returns test / train datasets

    train_set, test_set = [], []
    with open(path,'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data = list(lines)
        data_check_dups, data_fix =[], []

        for x in data:
            if x[:-1] not in data_check_dups:
                data_check_dups.append(x[:-1])
                data_fix.append(x)
        for x in data_fix:
            if random() < 0.5:
                train_set.append(x)
            else:
                test_set.append(x)

    rows_train = len(train_set)
    rows_test = len(test_set)
    csvfile.close()
    rows = min([rows_test, rows_train])
    write_data('sim_data/train.csv', train_set)
    write_data('sim_data/test.csv', test_set)
    train_set_fin = tf.data.experimental.make_csv_dataset('sim_data/train.csv', rows,
                                                    column_names=column_names,
                                                    label_name=label_name[0],
                                                    num_epochs=1,
                                                    shuffle=True)
    test_set_fin = tf.data.experimental.make_csv_dataset('sim_data/test.csv', rows,
                                                    column_names=column_names,
                                                    label_name=label_name[0],
                                                    num_epochs=1,
                                                    shuffle=True)
    return train_set_fin, test_set_fin, rows


def make_unknown_dataset(unknown_path, rows):
    # Reads the unknown data
    #length = 0
    with open(unknown_path, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data = list(lines)
        #length += len(data)
    csvfile.close()
    print(label_name[0])

    dataset = tf.data.experimental.make_csv_dataset(unknown_path, rows,  ##################
                                                    column_names=column_names,
                                                    label_name=label_name[0],
                                                    num_epochs=1,
                                                    shuffle=True)
    return dataset


def pack_features_vector(features, labels):

    features = tf.stack(list(features.values()), axis=1)
    return features, labels


#===============================================================================#


def make_model(train_dataset, rows, classnum, choice, metric_choice):

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))

    if choice == 'GHZ':
        if metric_choice == 'probabilities':

            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(250, activation='swish', input_shape=(rows, 6), kernel_regularizer = tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(150, activation='swish', kernel_regularizer = tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(75, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(classnum, kernel_regularizer = tf.keras.regularizers.l2(l=0.9))])
            return model, features, labels, train_dataset
            #raise
        elif metric_choice == 'fidelities':
            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(250, activation='swish', input_shape=(rows, 6), kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(150, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(75, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
            tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9))])
            return model, features, labels, train_dataset

    elif choice == 'Repeater':
        if metric_choice == 'probabilities':

            kreg = tf.keras.regularizers.l2(l=0.02)  # 0.08
            model = tf.keras.Sequential()
            tf.keras.layers.LayerNormalization(
                axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                beta_constraint=None, gamma_constraint=None, trainable=True, name=None)
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            return model, features, labels, train_dataset

        elif metric_choice == 'fidelities':
            kreg = tf.keras.regularizers.l2(l=0.2)  # 0.08
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(35, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(25, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(7, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            return model, features, labels, train_dataset

    elif choice == 'Teleportation':
        if metric_choice == 'probabilities':
            kreg = tf.keras.regularizers.l2(l=0.2)
            model = tf.keras.Sequential()
            tf.keras.layers.LayerNormalization(
                axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                beta_constraint=None, gamma_constraint=None, trainable=True, name=None
            )
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(24, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(24, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(24, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            return model , features, labels, train_dataset
        elif metric_choice == 'fidelities':

            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(250, activation='swish', input_shape=(rows, 6),
                                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(150, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(75, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9))])
            return model, features, labels, train_dataset

    elif choice == "WState":
        if metric_choice == 'probabilities':
            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(35, activation='swish', input_shape=(rows, 6),
                                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(25, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(7, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9))])
            return model, features, labels, train_dataset
            #raise
        elif metric_choice == 'fidelities':
            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(35, activation='swish', input_shape=(rows, 6),
                                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(25, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(7, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9))])
            """
            kreg = tf.keras.regularizers.l2(l=0.02)  # 0.08
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(256, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(32, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(16, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            """
            return model, features, labels, train_dataset
    elif choice == "Adder":
        if metric_choice == 'probabilities':

            kreg = tf.keras.regularizers.l2(l=0.2)  # 0.08
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            #model.add(tf.keras.layers.Dropout(.05))
            return model, features, labels, train_dataset

            #raise
        elif metric_choice == 'fidelities':
            kreg = tf.keras.regularizers.l2(l=0.2)  # 0.08
            model = tf.keras.Sequential()
            tf.keras.layers.LayerNormalization(
                axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                beta_constraint=None, gamma_constraint=None, trainable=True, name=None
            )
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            model.add(tf.keras.layers.Dense(64, activation='swish', activity_regularizer=kreg, input_shape=(rows, 8)))
            tf.keras.layers.Dropout(.05)
            return model, features, labels, train_dataset


#===============================================================================#


def loss(model, x, y):
    y = np.int_(y.numpy())
    tf.convert_to_tensor(y)
    y_ = model(x)
    loss_object = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_,)
    return loss_object


#===============================================================================#


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    grads = tape.gradient(loss_value, model.trainable_variables)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    return loss_value, grads


#===============================================================================#


def optimize(choice, metric_choice):
    # Params get tuned here, obviously some still in progress

    if choice == 'WState':
        if metric_choice == 'probabilities': # DONT TOUCH
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer
            #raise
        elif metric_choice == 'fidelities': #  DONNNT TOUUUCH
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.03, momentum=0.09, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')

            return optimizer

    elif choice == 'Teleportation':
        if metric_choice == 'probabilities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00095, decay=0.06, momentum=0.01, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer
            #raise
        elif metric_choice == 'fidelities': ######
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.03, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')

            return optimizer

    elif choice == 'GHZ':
        if metric_choice == 'probabilities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer
            #raise
        elif metric_choice == 'fidelities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer

    elif choice == 'Repeater':
        if metric_choice == 'probabilities': # DONT TOUCH
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False,centered=True, name='RMSProp')
            return optimizer
        elif metric_choice == 'fidelities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0007, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer

    elif choice == 'Adder':
        if metric_choice == 'probabilities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0001, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer
            #raise
        elif metric_choice == 'fidelities':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True, name='RMSProp')
            return optimizer

#===============================================================================#

def train_model(train_dataset,model,optimizer, num_epochs):

    train_loss_results, train_accuracy_results = [], []

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

def test_run(test_dataset, model):
    # Use known test data

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


def predict(test_dataset, model, mapper):
    # predict classes of unknown vectors

    test_dataset = test_dataset.map(pack_features_vector)
    predictions, predictions_converted = [], []

    for (x, y) in test_dataset:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        predictions.append(prediction.numpy().tolist())
    for x in range(len(predictions)):
        for y in range(len(predictions[x])):
            predictions_converted.append(mapper[str(int(predictions[x][y]))])

    return predictions_converted


#=========================Plotters etc =====================================#


def print_results(predictions, targets, mapper):
    for x in range(len(predictions)):
        for y in range(len(predictions[x])):
            print("_________")
            print("Predicted: ", mapper[str(int(predictions[x][y]))])
            print("Actual: ", mapper[str(int(targets[x][y]))])


def plotter(train_accuracy_results,train_loss_results):

    fig, axes = plt.subplots(2, sharex = True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()


#===============================================================================#

def build_and_run(train_dataset, test_dataset, unknown_dataset, mapper,
                  rows, batchsize, choice, predictions_total, metric_choice, num_epochs):

    model, features, labels, train_dataset = make_model(train_dataset, rows, batchsize, choice, metric_choice)
    grad(model, features, labels)
    optimizer = optimize(choice, metric_choice)

    train_model(train_dataset, model, optimizer, num_epochs)

    #plotter(acc_results, loss_results)
    predictions, targets = test_run(test_dataset, model)
    #print_results(predictions, targets, mapper)

    # saves the predictions on the unknown datasets over multiple calls of build_and_run
    # so the final answer is an average
    predictions_converted = predict(unknown_dataset, model, mapper)
    predictions_total.append(predictions_converted)


def designate_predictions(predictions_total):
    # Find the overall frequency of class predictions on the unknown data

    print("Predicting Unknowns....")
    count = 0
    frequencies = {}
    for x in predictions_total:
        for y in x:
            if y not in frequencies.keys():
                frequencies[y] = 1
                count += 1
            else:
                frequencies[y] += 1
                count += 1
    for x in frequencies.keys():
        print(x, ":", frequencies[x] / count)

    return

#===============================================================================#

# Placement lists

metric_choices = ['probabilities', 'fidelities']
choices = ["Teleportation", "WState", "GHZ", "Repeater", "Adder"]
#batchsizes_probs = [60, 60, 60, 80, 60]
#batchsizes_fids = [60, 100, 20, 20, 20]
batchsizes_probs = [100, 60, 60, 60, 100]
batchsizes_fids = [100, 100, 60, 60, 60]
epochs_probs = [1000, 2000, 600, 600, 600]
epochs_fids = [1700, 4000, 600, 600, 600] # check all these
batchsizes_choice = [batchsizes_probs, batchsizes_fids]
epochs_choice = [epochs_probs, epochs_fids]

def main(path, choice, unknown_path, metric_choice):

    cnames = [column_names_list_prob[choices.index(choice)], column_names_list_fid[choices.index(choice)]]
    batchsize_list = batchsizes_choice[metric_choices.index(metric_choice)]
    batchsize = batchsize_list[choices.index(choice)]
    num_epochs = epochs_choice[metric_choices.index(metric_choice)][choices.index(choice)]
    mapper = maps[choices.index(choice)]
    column_names_temp = cnames[metric_choices.index(metric_choice)]
    count = 0
    for x in column_names_temp:
        column_names.append(x)
        if count < len(column_names_temp) - 1:
            feature_names.append(x)
        else:
            label_name.append(x)
        count += 1
    print(column_names_temp)
    train_dataset, test_dataset, rows = make_dataset(path)
    unknown_dataset = make_unknown_dataset(unknown_path, rows)

    predictions_total = []
    # predictions on unknowns append to predictions_total
    for x in range(3):
        build_and_run(train_dataset, test_dataset, unknown_dataset, mapper,
                      rows, batchsize, choice, predictions_total, metric_choice, num_epochs)

    # predict which errors are present in unknown data from the 3 runs combined
    designate_predictions(predictions_total)

    # remove the temporary file needed to avoid using split_data_files
    # this does not happen till the very end
    os.remove('sim_data/train.csv')
    os.remove('sim_data/test.csv')



unknown_path = "ibm_data/repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv"
path = "sim_data/repeater_600fidelities_.csv"
main(path, 'Repeater', unknown_path, 'fidelities')


""" Results 7. 25. 20 
"WState" (fidelities) wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; wstate_100fidelities_.csv 
Run1: Train Accuracy: 77.380% ; Test set accuracy: 76.410%
Run2: Train Accuracy: 76.653% ; Test set accuracy: 74.894%  DONE 
Run3: Train Accuracy: 75.500% ; Test set accuracy: 75.076% 
Predicting Unknowns....
CNOT : 0.9494949494949495
RY : 0.04377104377104377
X : 0.006734006734006734
"WState" (probabilities) "ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/wstate_300probabilities_.csv" 
Run1: Train Accuracy: 97.452% ; Test set accuracy: 97.217%
Run2: Train Accuracy: 96.448% ; Test set accuracy: 95.898%  DONE 
Run3: Train Accuracy: 96.803% ; Test set accuracy: 93.723% 
Predicting Unknowns....
CNOT : 0.5050505050505051
X : 0.3367003367003367
RY : 0.15824915824915825

"GHZ" (fidelities) "ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv" ; "sim_data/ghz_300fidelities_.csv"
Run1: Train Accuracy: 99.669% ; Test set accuracy: 94.604%
Run2: Train Accuracy: 99.669% ; Test set accuracy: 99.831% DONE 
Run3: Train Accuracy: 99.669% ; Test set accuracy: 99.494%
Predicting Unknowns....
CNOT : 1.0
"GHZ" (probabilities) "ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/ghz_300probabilities_.csv"
Run1: Train Accuracy: 96.713% ; Test set accuracy: 96.774%
Run2: Train Accuracy: 95.484% ; Test set accuracy: 91.869% DONE 
Run3: Train Accuracy: 96.713% ; Test set accuracy: 96.774%
Predicting Unknowns....
HADAMARD : 1.0


"Repeater" (probabilities) "ibm_data/repeater100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/repeater_300probabilities_.csv"
Run1: Train Accuracy: 91.361% ; Test set accuracy: 82.585%
Run2: Train Accuracy: 95.458% ; Test set accuracy: 83.730% DONE
Run3: Train Accuracy: 93.218% ; Test set accuracy: 91.242%
Predicting Unknowns....
CNOT : 0.9932659932659933
HADAMARD : 0.006734006734006734
"Repeater" (fidelities) "ibm_data/repeater100_ibm_sim_fidelitiesibmq_16_melbourne.csv" ; "repeater_600fidelities_.csv"
Run1: 69.085% ; Test set accuracy: 61.082%  # Have not found a better way for repeater fid.
Run2: 66.969% ; Test set accuracy: 62.480%  
Run3: 69.448% ; Test set accuracy: 60.677%
Predicting Unknowns....
HADAMARD : 0.9966499162479062
CNOT : 0.0033500837520938024


Adder(fidelities) "ibm_data/adder100_ibm_sim_fidelitiesibmq_16_melbourne.csv"; "sim_data/adder_300fidelities_.csv"
Run1: Train Accuracy: 93.610% ; Test set accuracy: 93.598%
Run2: Train Accuracy: 91.704% ; Test set accuracy: 93.598%  # Check for accidentally changed parameters
Run3: Train Accuracy: 94.058% ; Test set accuracy: 94.150%
Predicting Unknowns....
CNOT : 1.0
"Adder" (probabilities) "ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv"; "sim_data/adder_300probabilities_.csv"
# Adder will probably be altered to break down the toffolis into classes
# Since a toffoli is full of cnots..
Run1: Train Accuracy: 94.609% ; Test set accuracy: 93.813% 
Run2: Train Accuracy: 93.509% ; Test set accuracy: 94.488%  # Check for accidentally changed parameters 
Run3: Train Accuracy: 93.069% ; Test set accuracy: 93.251% 
Predicting Unknowns....
CNOT : 0.9797979797979798
TOFFOLI : 0.020202020202020204


"Teleportation" (fidelities) teleport100_ibm_sim_fidelitiesibmq_16_melbourne.csv ; teleport_100fidelities_.csv 
Run1: Train Accuracy: 81.283% ; Test set accuracy: 78.682%
Run2: Train Accuracy: 85.625% ; Test set accuracy: 58.024%  # Check for accidentally changed parameters
Run3: Train Accuracy: 84.563% ; Test set accuracy: 82.871%
Predicting Unknowns....
CNOT : 0.9461279461279462
HADAMARD : 0.05387205387205387
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"Teleportation" (probabilities) teleport100_ibm_sim_probabilities_ibmq_16_melbourne.csv ; teleport_100probabilities_.csv
Run1: Train Accuracy: 85.687% ; Test set accuracy: 84.295%
Run2: Train Accuracy: 85.879% ; Test set accuracy: 84.295%  # Check for accidentally changed parameters
Run3: Train Accuracy: 84.918% ; Test set accuracy: 83.728%
Predicting Unknowns....
HADAMARD : 0.8013468013468014
CNOT : 0.19865319865319866
"""


"""Optimizers for tuning more stuff later 
#optimizer = tf.keras.optimizers.Nadam(learning_rate=.09, epsilon=1e-12, beta_1=0.999, beta_2=0.99999)
#optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.0, epsilon=1e-10, use_locking=False,centered=True, name='RMSProp')
#optimizer = tf.compat.v1.train.MomentumOptimizer(
#   0.6, 0.9, use_locking=False, name='Momentum', use_nesterov=True
#)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=1, nesterov=True)
"""
