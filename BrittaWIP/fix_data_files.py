import csv
"""
:Code to modify class columns of csv files for numeric values, 
:and record a dictionary mapping classes to numbers. 
:Can change files list as needed
"""
FILES = ['2RepeaterTrainingData200Had_Indexed.csv', '2RepeaterTrainingData200new_Indexed.csv',
         '2RepeaterTrainingData200QFT2_Indexed.csv','2RepeaterTrainingData200QFT_Indexed.csv',
         'GHZTrainingData200Had_Indexed.csv','GHZTrainingData200new_Indexed.csv',
         'GHZTrainingData200QFT2.csv','GHZTrainingData200QFT_Indexed.csv','RepeaterTrainingData20Had_Indexed.csv',
         'RepeaterTrainingData20QFT2_Indexed.csv','RepeaterTrainingData20QFT_Indexed.csv',
         'RepeaterTrainingData200Had_Indexed.csv','RepeaterTrainingData200new_Indexed.csv',
         'RepeaterTrainingData200QFT2_Indexed.csv','RepeaterTrainingData200QFT_Indexed.csv',
         'RepeaterTrainingDataHad_Indexed.csv','RepeaterTrainingDataQFT2_Indexed.csv',
         'RepeaterTrainingDataQFT_Indexed.csv','TeleportTrainingData20Had_Indexed.csv',
         'TeleportTrainingData20new_Indexed.csv','TeleportTrainingData20QFT2_Indexed.csv',
         'TeleportTrainingData20QFT_Indexed.csv','TeleportTrainingData200Had_Indexed.csv',
         'TeleportTrainingData200Had_Indexed.csv','TeleportTrainingData200QFT2_Indexed.csv',
         'TeleportTrainingData200QFT_Indexed.csv']

SAVEPLACE = 'training_data_with_numerals/'

for path in FILES:
    CATS, dex, DATA_ALT = {}, 0, []
    with open('TrainingData (copy)/'+path, 'r') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        for r in data:
            if len(r) == 6 and r[-2:] not in CATS:
                CATS[r[-2:]] = dex
                dex += 1
            if len(r) == 5 and r[-1] not in CATS:
                CATS[r[-1]] = dex
                dex += 1
        DATA_ALT = [line[:-1] + [CATS[line[-1]]] for line in data]
    csvfile.close()

    with open(SAVEPLACE + path[:-4] + '_num.csv', 'a', newline='') as csvFile:
        for line in DATA_ALT:
            writer = csv.writer(csvFile)
            writer.writerow(line)
        csvFile.close()

    with open(SAVEPLACE + path[:-4] + '_cats.txt', 'a', newline='') as csvFile:
        for key in CATS.keys():
            writer = csv.writer(csvFile)
            writer.writerow([key+':  '+str(CATS[key])])
        csvFile.close()
