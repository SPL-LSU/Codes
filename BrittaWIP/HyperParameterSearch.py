# Code which seeks out optimal hyperparameters for a neural network
# Was an experiment since it did not seem to go well
# B Manifold July
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

data = pd.read_csv('wstate_40.csv')
data.head(20)

train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=50):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Gate')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 100
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=True, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=True, batch_size=batch_size)

# numarical features
num_c = ['State1', 'State2', 'State3', 'State4', 'State5', 'State6', 'State7', 'State8']


def get_scal(feature):
    def minmax(x):
        mini = train[feature].min()
        maxi = train[feature].max()

        return (x - mini) / (maxi - mini)

    return (minmax)


# Numerical columns
feature_columns = []
for header in num_c:
    scal_input_fn = get_scal(header)
    feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn))

len(feature_columns)

HP_NUM_UNITS1 = hp.HParam('num_units 1', hp.Discrete([4, 500]))
HP_NUM_UNITS2 = hp.HParam('num_units 2', hp.Discrete([4, 500]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.000009, 0.01))
HP_LEARN_RATE = hp.HParam('learn_rate', hp.RealInterval(0.00009, 0.1))
HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.0001, .1))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_DROPOUT, HP_L2, HP_LEARN_RATE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

from tensorflow.keras.optimizers import Nadam


def train_test_model(hparams):
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish'),
        layers.Dropout(hparams[HP_DROPOUT]),
        layers.Dense(hparams[HP_NUM_UNITS2], kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish'),
        layers.Dense(1, activation='softmax')
    ])

    opt = tf.keras.optimizers.Nadam(lr=hparams[HP_LEARN_RATE], epsilon=1e-12, beta_1=0.999, beta_2=0.99999)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5)
    _, accuracy = model.evaluate(val_ds)
    return accuracy


import random


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0

for x in range(100):
    num_units1 = random.choice(HP_NUM_UNITS1.domain.values)
    num_units2 = random.choice(HP_NUM_UNITS2.domain.values)
    dropout_rate = random.uniform(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value)
    l2 = random.uniform(HP_L2.domain.min_value, HP_L2.domain.max_value)
    learn_rate = random.uniform(HP_LEARN_RATE.domain.min_value, HP_LEARN_RATE.domain.max_value)

    hparams = {
        HP_NUM_UNITS1: num_units1,
        HP_NUM_UNITS2: num_units2,
        HP_DROPOUT: dropout_rate,
        HP_L2: l2,
        HP_LEARN_RATE: learn_rate

    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1
