"""
BNM Jun 23 '20

Code to split up data files into two equal train/test files
for tensorflow neural net

"""


from random import shuffle, random
import csv
import numpy as np


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



data_loc = "training_data_with_numerals/June22/June24TeleportTrainingData400QFT2.csv"
divide_csvfile(data_loc)
