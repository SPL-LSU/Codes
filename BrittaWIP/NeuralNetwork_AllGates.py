# Neural network testing with IBM data
# Same as other, but attempting to be able to compare with KNN results on index-based data (like original code)
# 7.25 results at bottom

# Has to write and then delete temp, but won't succeed if stopped early. This is to not need split_data_files.py
# ValueError: Problem inferring types: CSV row has different number of fields than expected. This means there is a problem with column_names
# or that the wrong circuit name was given to main()


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


REPEATER_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"HADAMARD2", '2':"CNOT1", '3':"CNOT2", '4':"HADAMARD3", '5':"HADAMARD4", '6':"HADAMARD5",
                             '7':"HADAMARD6", '8':"CNOT3", '9':"HADAMARD7", '10':"HADAMARD8", '11':"HADAMARD9", '12':"CNOT4", '13':"HADAMARD10",
                             '14':"HADAMARD11", '15':"CNOT5", '16':"HADAMARD12", '17':"HADAMARD13", '18':"CNOT6", '19':"HADAMARD14", '20':"HADAMARD15",
                             '21':"CNOT7", '22':"HADAMARD16", '23':"CNOT8", '24':"HADAMARD17", '25':"HADAMARD18", '26':"CNOT9", '27':"HADAMARD19",
                             '28':"HADAMARD20", '29':"CNOT10", '30':"HADAMARD21", '31':"HADAMARD22", '32':"CNOT11"}
GHZ_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"CNOT1", '2':"CNOT2", '3':"CNOT3"}
TELEPORT_TRAINING_MAP_ALL = {'0':"HADAMARD1", '1':"HADAMARD2", '2':"CNOT1", '3':"CNOT2", '4':"HADAMARD3", '5':"CNOT3", '6':"CZ1"}
W_STATE_TRAINING_MAP_ALL = {'0':"RY1", '1':"X1", '2':"X2", '3':"CNOT1", '4':"RY2", '5':"CNOT2", '6':"RY3", '7':"X3", '8':"X4", '9':"CNOT3", '10':"CNOT4"}
ADDER_TRAINING_MAP_ALL = {'0':"TOFFOLI1", '1':"TOFFOLI2", '2':"TOFFOLI3", '3':"CNOT1", '4':"CNOT2", '5':"CNOT3"}

maps = [TELEPORT_TRAINING_MAP_ALL, W_STATE_TRAINING_MAP_ALL, GHZ_TRAINING_MAP_ALL, REPEATER_TRAINING_MAP_ALL, ADDER_TRAINING_MAP_ALL]


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

# Add the class column to recognize the data
for x in column_names_list_prob:
    x.append("Gate")
for x in column_names_list_fid:
    x.append("Gate")
# These are awkwardly filled out in main()
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
    kreg = tf.keras.regularizers.l2(l=0.02)
    normalize_layer = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=0.0001, center=True, scale=True, beta_initializer='zeros',
        gamma_initializer='ones', beta_regularizer=kreg, gamma_regularizer=kreg, trainable=True)

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
            # STILL NEED
            kreg = tf.keras.regularizers.l2(l=0.9)
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(normalize_layer)
            model.add(tf.keras.layers.Dense(8, activation='swish', activity_regularizer=kreg, input_shape=(rows, 16)))
            model.add(tf.keras.layers.Dense(8, activation='swish', activity_regularizer=kreg))
            model.add(tf.keras.layers.Dense(8, activation='swish', activity_regularizer=kreg))
            model.add(tf.keras.layers.Dense(33, activation='swish', activity_regularizer=kreg))
            return model, features, labels, train_dataset

        elif metric_choice == 'fidelities': # STILL NEED
            kreg = tf.keras.regularizers.l2(l=0.9)
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(normalize_layer)
            model.add(tf.keras.layers.Dense(10, activation='swish', activity_regularizer=kreg, input_shape=(rows, 4)))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(10, activation='swish', activity_regularizer=kreg))
            #model.add(tf.keras.layers.Dropout(.05))
            model.add(tf.keras.layers.Dense(10, activation='swish', activity_regularizer=kreg))
            model.add(tf.keras.layers.Dense(33, activation='swish', activity_regularizer=kreg))
            return model, features, labels, train_dataset

    elif choice == 'Teleportation': # STILL NEED
        if metric_choice == 'probabilities':
            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(250, activation='swish', input_shape=(rows, 8),
                                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(150, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(75, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)),
                                         tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9))])
            return model, features, labels, train_dataset
        elif metric_choice == 'fidelities': # STILL NEED

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(250, activation='swish', input_shape=(rows, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.9)))
            model.add(tf.keras.layers.Dense(75, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.9)))
            model.add(tf.keras.layers.Dense(classnum, kernel_regularizer=tf.keras.regularizers.l2(l=0.9)))
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
        elif metric_choice == 'fidelities':
            model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(35, activation='swish', input_shape=(rows, 4),
                                                               activity_regularizer=tf.keras.regularizers.l2(l=0.09)),
                                         tf.keras.layers.Dense(15, activation='selu', activity_regularizer=tf.keras.regularizers.l2(l=0.09)),
                                         tf.keras.layers.Dense(15, activation='swish', activity_regularizer=tf.keras.regularizers.l2(l=0.09)),
                                         tf.keras.layers.Dense(classnum, activity_regularizer=tf.keras.regularizers.l2(l=0.09))])
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
        if metric_choice == 'probabilities':
            return tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)
        elif metric_choice == 'fidelities':
            return tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.06, momentum=.3, epsilon=1e-10, use_locking=False, centered=True)

    elif choice == 'Teleportation':
        if metric_choice == 'probabilities': # STILL NEED
            return tf.keras.optimizers.SGD(learning_rate=0.002, momentum=.1, nesterov=True)
            #tf.compat.v1.train.RMSPropOptimizer(0.00015, decay=0.01, momentum=0.05, epsilon=1e-10, use_locking=False, centered=True)
        elif metric_choice == 'fidelities': # STILL NEED
            return tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.03, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)

    elif choice == 'GHZ':
        if metric_choice == 'probabilities':
            return tf.compat.v1.train.RMSPropOptimizer(0.0009, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)
        elif metric_choice == 'fidelities':
            return tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)

    elif choice == 'Repeater':
        if metric_choice == 'probabilities': # STILL NEED
            return tf.compat.v1.train.RMSPropOptimizer(0.00009, decay=0.01, momentum=.1, epsilon=1e-10, use_locking=False, centered=True)
        elif metric_choice == 'fidelities': # STILL NEED
            #return tf.compat.v1.train.RMSPropOptimizer(0.0001, decay=0.01, momentum=0.01, epsilon=1e-10, use_locking=False, centered=True)
            return tf.compat.v1.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam')

    elif choice == 'Adder':
        if metric_choice == 'probabilities':
            return tf.compat.v1.train.RMSPropOptimizer(0.0001, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)
        elif metric_choice == 'fidelities':
            return tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.1, epsilon=1e-10, use_locking=False, centered=True)


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
batchsizes_probs = [100, 60, 60, 60, 100]
batchsizes_fids = [100, 100, 60, 60, 60]
epochs_probs = [4000, 10000, 1000, 10000, 1000] # Dont know w-state probs epochs anymore
epochs_fids = [4000, 4000, 1000, 4000, 1000]
batchsizes_choice = [batchsizes_probs, batchsizes_fids]
epochs_choice = [epochs_probs, epochs_fids]

def main(path, choice, unknown_path, metric_choice):

    cnames = [column_names_list_prob[choices.index(choice)], column_names_list_fid[choices.index(choice)]]
    batchsize_list = batchsizes_choice[metric_choices.index(metric_choice)]
    batchsize = batchsize_list[choices.index(choice)]
    num_epochs = epochs_choice[metric_choices.index(metric_choice)][choices.index(choice)]
    print(num_epochs)
    mapper = maps[choices.index(choice)]
    print(mapper)
    column_names_temp = cnames[metric_choices.index(metric_choice)]
    print(column_names_temp)
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
    for x in range(1):
        build_and_run(train_dataset, test_dataset, unknown_dataset, mapper,
                      rows, batchsize, choice, predictions_total, metric_choice, num_epochs)

    # predict which errors are present in unknown data from the 3 runs combined
    designate_predictions(predictions_total)

    # remove the temporary file needed to avoid using split_data_files
    # this does not happen till the very end
    os.remove('sim_data/train.csv')
    os.remove('sim_data/test.csv')



unknown_path = "ibm_data/repeater100_ibm_sim_probabilities_ibmq_16_melbourne.csv"
path = "sim_data/repeater_800probabilities_allgates_.csv"
main(path, 'Repeater', unknown_path, 'probabilities')


""" Results 7. 25. 20 
# ===================================================================================
"WState" (probabilities) "ibm_data/wstate100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/wstate_800probabilities_allgates_.csv"
Run1: 93.411% ; Test set accuracy: 93.066%
Run2: 88.315% ; Test set accuracy: 88.158%
Run3: 93.549% ; Test set accuracy: 92.616% 
Predictions: 
Predicting Unknowns....
RY2 : 0.18181818181818182
X3 : 0.468013468013468
X1 : 0.016835016835016835
CNOT4 : 0.28619528619528617
CNOT2 : 0.010101010101010102
X2 : 0.016835016835016835
CNOT3 : 0.006734006734006734
X4 : 0.013468013468013467
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"WState" (fidelities) "ibm_data/wstate100_ibm_sim_fidelitiesibmq_16_melbourne.csv" ; "sim_data/wstate_800fidelities_allgates_.csv"
Run1: 72.529% ; Test set accuracy: 71.474% 
Run2: 73.755% ; Test set accuracy: 72.146%
Run3: 72.938% ; Test set accuracy: 71.611%
Predictions: 
Predicting Unknowns....
CNOT3 : 0.4444444444444444
CNOT4 : 0.40404040404040403
X2 : 0.04713804713804714
RY1 : 0.0707070707070707
CNOT2 : 0.03367003367003367
# ===================================================================================
"GHZ" (fidelities)  "ibm_data/ghz100_ibm_sim_fidelitiesibmq_16_melbourne.csv" ; "sim_data/ghz_800fidelities_allgates_.csv" 
Run1: 97.910% ; Test set accuracy: 86.633%
Run2: 84.942% ; Test set accuracy: 84.341%
Run3: 96.558% ; Test set accuracy: 96.308%
Predictions: 
Predicting Unknowns....
CNOT1 : 0.48148148148148145
CNOT3 : 0.5185185185185185
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"GHZ" (probabilities)   "ibm_data/ghz100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/ghz_800probabilities_allgates_.csv"
Run1: 93.829% ; Test set accuracy: 86.087%
Run2: 99.876% ; Test set accuracy: 94.207% 
Run3: 99.937% ; Test set accuracy: 96.646%
Predictions: 
Predicting Unknowns....
HADAMARD1 : 0.9932659932659933
CNOT1 : 0.006734006734006734
# ===================================================================================
# Repeater and Teleport are not doable with this code
# ===================================================================================
Adder(fidelities) 
Run1:  89.057% ; Test set accuracy: 88.479% 
Run2:  88.973% ; Test set accuracy: 89.557%
Run3:  85.827% ; Test set accuracy: 85.535%
Predictions: 
Predicting Unknowns....
CNOT1 : 0.003367003367003367
CNOT2 : 0.9966329966329966
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"Adder" (probabilities) "ibm_data/adder100_ibm_sim_probabilities_ibmq_16_melbourne.csv" ; "sim_data/adder_800probabilities_allgates_.csv"
Run1: 98.326% ; Test set accuracy: 97.094%
Run2: 98.326% ; Test set accuracy: 96.887% 
Run3: 98.409% ; Test set accuracy: 96.762% 
Predictions: 
Predicting Unknowns....
CNOT3 : 0.45791245791245794
TOFFOLI3 : 0.026936026936026935
CNOT2 : 0.12121212121212122
CNOT1 : 0.3569023569023569
TOFFOLI2 : 0.037037037037037035
# ===================================================================================


"""
"""Optimizers for tuning more stuff later 
#optimizer = tf.keras.optimizers.Nadam(learning_rate=.09, epsilon=1e-12, beta_1=0.999, beta_2=0.99999)
#optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005, decay=0.06, momentum=0.0, epsilon=1e-10, use_locking=False,centered=True, name='RMSProp')
#optimizer = tf.compat.v1.train.MomentumOptimizer(
#   0.6, 0.9, use_locking=False, name='Momentum', use_nesterov=True
#)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=1, nesterov=True)
"""