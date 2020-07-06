"""
BNM Jun 23 '20

Code to split up data files into two equal train/test files
for tensorflow neural net


"""


from random import shuffle, random
import csv
import numpy as np



def balance_frequency(list):

    classes = [x[-1] for x in list]
    minimum_class = min(set(classes), key=classes.count)
    min_freq = 0

    for x in list:
        if x[-1] == minimum_class:
            min_freq += 1
    class_counts = [0]*len(set(classes))
    class_set = np.unique(classes).tolist()
    new_list = []

    for x in range(len(list)):
        temp_class = list[x][-1]
        index = class_set.index(temp_class)
        if class_counts[index] <= min_freq:
            new_list.append(list[x])
            class_counts[index] += 1

    fixed_classes = [x[-1] for x in new_list]
    freq_check = []
    for x in np.unique(fixed_classes).tolist():
        count = 0
        for y in fixed_classes:
            if y == x:
                count += 1
        freq_check.append(count)

    print(freq_check)

    return new_list

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


        for x in data_fix:

            if random() < 0.5:
                train_set.append(x)
            else:
                test_set.append(x)

    train_set = balance_frequency(train_set)
    test_set = balance_frequency(test_set)
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



data_loc = "training_data_with_numerals/WTrainingData600QFT2.csv"

divide_csvfile(data_loc)
""" bit to write some unsplit data balanced freq
with open(data_loc, 'r') as csvfile:
    lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    data = list(lines)
    data_check_dups, data_fix = [], []
    for x in data:
        if x[:-1] not in data_check_dups:
            data_check_dups.append(x[:-1])
            data_fix.append(x)

dataset = balance_frequency(data_fix)
csvfile.close()

with open(data_loc[:-4] + "_.csv", 'a', newline='') as csvFile:
    for p in dataset:
        writer = csv.writer(csvFile)
        writer.writerow(p)
    csvFile.close()
"""
