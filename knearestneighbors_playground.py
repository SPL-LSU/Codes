# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:21:18 2019

Set of functions used to perform k nearest-neighbors.

Instructions for use:
    find csv file location: replace it in handleDataset, being careful to use / instead of \
    change the split in handleDataset, second input, as needed
    uncomment print statements as necessary    

@author: Margarite L. LaBorde
@email:  mlabo15@lsu.edu
"""
import numpy as np
import qutip as qt
import operator
import csv
import math
import random

#splits the data set into testing and training data. Need to initialize empy vectors first
def handleDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(len(dataset[0])-1):
                dataset[x][y]=float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
        csvfile.close()
    return 0

#Finds the euclidean distance bewteen two vectors of length 'length;
def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)

#finds the k points nearest a test point
def getKNeighbors(trainingSet,test,k,tolerance,idealvector):
    distances=[]
    length=len(test)-1
    neighbors=[]
    prob=[]
    for i in range(length):
        prob.append(test[i])
    if within_tolerance(tolerance,prob,idealvector):
        #override
        for x in range(k):
            neighbors.append([0,0,0,0,"tolerance"])
    else:
        for x in range(len(trainingSet)):
            dist=euclideanDistance(test,trainingSet[x],length)
            distances.append((trainingSet[x],dist))
        distances.sort(key=operator.itemgetter(1))
        for x in range(k):
            neighbors.append(distances[x][0])
    return neighbors

#determines the classes of a vector of neighbors and returns a prediction of a test point's class
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
#Finds the accuracy of a test set prediction   
def getAccuracy(testSet,predictions):
    correct=0
    magnitude=float(len(testSet))
    for x in range(len(testSet)):
        if predictions[x] is "tolerance":
            magnitude-=1
        if testSet[x][-1] is predictions[x]:
            correct+=1
    accuracy=(correct/magnitude)*100.0
    return accuracy

def within_tolerance(tolerance,probvector,idealvector):
    length=4
    scale=1-tolerance
    test=scale*euclideanDistance([0,0,0,0],idealvector,length)
    dist=euclideanDistance(probvector,idealvector,length)
    if dist <= test:
        truth = True
    else:
        truth = False
    return truth

def main():
    testingset=[]
    trainingset=[]
    
    #Insert csv file name and split here:
    path=input("Give csv file location, remembering to use forward slashes: ")
    split=input("Give training set split, in the form of a number between 0 and 1: ")
    split=float(split)
    if split > 1 or split < 0:
        print("Incorrect split input given. Default used.")
        split = 0.66
    handleDataset(path,split,trainingset,testingset)

    print ('Train set: ' + repr(len(trainingset)) )
    print ('Test set: ' + repr(len(testingset)) )
    
    # generate predictions 
    predictions=[] 
    k =input("Give a k value: ")
    k=int(k)
    tolerance=input("Give a tolerance: ")
    tolerance=float(tolerance)
    ideal=[0.7437184400574876, 0.7437184400574876, 0.7437184400574876, 0.5625000046098119]
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset, testingset[x], k,tolerance,ideal) 
        result = getResponse(neighbors) 
        predictions.append(result) 
        print('> predicted=' + repr(result) + ', actual=' + repr(testingset[x][-1]))
        
    accuracy = getAccuracy(testingset, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    csvData = [[repr(result),repr(testingset[x][-1])]]
             
    with open('accuracy_data.csv','a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close
    return 0

main()