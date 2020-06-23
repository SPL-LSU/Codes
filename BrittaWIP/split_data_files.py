"""
BNM Jun 23 '20

Code to split up data files into two equal train/test files
for tensorflow neural net

"""


from random import shuffle, random
import csv
import numpy as np


def robust_scale(set):
    """
    Rescale the dataset to fight outliers
    """
    n_features = len(set[0]) - 1
    arr = np.array(set)
    q_list = []

    if True == True:
        for x in range(n_features):
            min_i, max_i = np.min(arr[:, x]), np.max(arr[:, x])
            median = (max_i - min_i) / 2
            q3 = (max_i - median) / 2 + median
            q1 = (max_i - median) / 2
            q_list.append((q1, q3))

            for y in range(len(set)):
                arr[y, x] = (arr[y, x] - q1) / (q3 - q1)
        new_set = arr.tolist()
        return new_set


def balance_frequency(set):
    """
    Ensures equal frequency of all class vectors
    """

    cats, new_set = [], []
    for x in set:
        cats.append(x[-1])

    max_cat = int(max(cats))
    cat_counts = [0]*max_cat

    for x in range(max_cat):
        for y in set:
            if int(y[-1]) == x:
                cat_counts[x] += 1

    min_cat_counts = min(cat_counts)
    freqs = [0]*max_cat

    for x in range(len(set)):
        cat = int(set[x][-1])
        if cat < max_cat:
            if freqs[cat] <= min_cat_counts:
                new_set.append(set[x])

            freqs[cat] += 1

    return new_set

#===============================================================================#

def divide_csvfile(path):
    """
    Splits data into test/train
    Writes training/test vectors into new file
    """
    train_set,test_set,train_cats,test_cats = [],[],[],[]

    with open(path,'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data = list(lines)
        data_check_dups, data_fix =[], []
        for x in data:
            if x[:-1] not in data_check_dups:
                data_check_dups.append(x[:-1])
                data_fix.append(x)

        data_fix = robust_scale(data_fix)

        for x in data_fix:

            if random() < 0.5:
                train_set.append(x)
            else:
                test_set.append(x)

    train_set = balance_frequency(train_set)
    test_set = balance_frequency(train_set)
    csvfile.close()

    with open(path[:-4] + "_train.csv", 'a', newline='') as csvFile:
        for p in train_set:
            writer = csv.writer(csvFile)
            writer.writerow(p)
        csvFile.close()

    with open(path[:-4] + "_test.csv", 'a', newline='') as csvFile:
        for p in test_set:
            writer = csv.writer(csvFile)
            writer.writerow(p)
        csvFile.close()

    return train_set, test_set



data_loc = "training_data_with_numerals/June21/GHZTrainingDataHad2_June21_num.csv"
divide_csvfile(data_loc)