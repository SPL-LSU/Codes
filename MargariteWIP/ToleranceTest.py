# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:05:41 2020

@author: marga
"""

import numpy as np
import qutip as qt
import math
import csv
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
"""
KNN Block
"""

#splits the data set into testing and training data. Need to initialize empty vectors first
def handleDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(len(dataset[0])-1):
                dataset[x][y]=float(dataset[x][y])
                if random() < split:
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
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
#Finds the accuracy of a test set prediction   
def getAccuracy(testSet, predictions):
    correct=0
    tol=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        if testSet[x][-1] != predictions[x]:
            if predictions[x] == "tolerance":
                tol+=1
            #print("predicted:" + predictions[x],"actual:" + testSet[x][-1])
    total_length=float(len(testSet))-tol
    print("#Within Tolerance: ")
    print(tol)
    return (correct/total_length)*100.0
    
def KNN(path,split,k,tolerance,idealvector):
    testingset=[]
    trainingset=[]
    
    #Insert csv file name and split here:
    #path=input("Give csv file location, remembering to use forward slashes: ")
    split = split
    split=float(split)
    if split > 1 or split < 0:
        print("Incorrect split input given. Default used.")
        split = 0.66
    handleDataset(path,split,trainingset,testingset)
    
    # generate predictions 
    predictions=[] 
    k=k
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k,tolerance,idealvector) 
        result = getResponse(neighbors) 
        predictions.append(result) 
        #print('> predicted=' + repr(result) + ',actual=' + repr(testingset[x][-1]))
        
    accuracy = getAccuracy(testingset,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
###############################################################################
"""
Basic building-block functions for classification procedure
"""

#Fixes qutip's weird dimension problems
def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

id2=qt.Qobj(np.identity(2))

#applies an array of gates (circuit) and applies it to a state
def basic_b(state,array):
    for i in range(len(array)):
        if type(array[i]) is str:
            continue
        else:
            k=qt.Qobj(array[i])
            #print(type(k))
            k=tensor_fix(k)
            #print(k.type,k.shape)
            state=qt.Qobj(state)
            #print(state)
            state=k(state)
    return state
    
#Fidelity distance metric
def dis(state1,state2):
    (n,m)=state1.shape
    if state1.type is not 'ket' or state2.type is not 'ket':
        print("Error: One or more input to the distance function is not a ket." )
        p_0=0
    else:
        fid=qt.fidelity(state1,state2)
        p_0 = 1/n + (n-1)*fid/n
    return p_0

def gate_troubleshooter(gate,n):
    if gate.shape != (n,n):
        gate=tensor_fix(gate)
        while gate.shape !=(n,n):
            seed=randint(0,1)
            if seed ==1:
                gate=qt.tensor(id2,gate)
                gate=tensor_fix(gate)
            else:
                gate=qt.tensor(gate,id2)
                gate=tensor_fix(gate)
        if gate.shape != (n,n):
            print("FUUUUUUUUUUUUUUUUU")
        #else:
            #print("fixed it!")  
    return gate

"""
Hadamard generation gates
"""


#affect is list of affected qubits, 0 indexed
#makes an n qubit hadamard, affecting a subset of qubits
def hadamaker(qubits,affected):
    array=[]
    i=0
    while len(array) < qubits:
        if i in affected:
            gate=qt.hadamard_transform()
            gate=tensor_fix(gate)
        else:
            gate=id2
        array.append(gate)
        i+=1
    hadamade=qt.Qobj([[1]])
    for gate in array:
        hadamade = qt.tensor(hadamade,gate)
        hadamade=tensor_fix(hadamade)
    return hadamade

"""
Random Unitary gates/functions
"""
#code which give an original unitary gate
def random_unitary_gate(delta,alpha,theta,beta,value):
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate

def random_angles():
    #gets a random value for each variable in the gate
    choice = randint(1,4)
    unitary_gate = ()
    if choice == 1: #Pauli-Y Gate
        unitary_gate = (0.0,math.pi/2,2*math.pi,math.pi/2,True)
    elif choice == 2: #Pauli-Z Gate
        unitary_gate = (0.0,0.0,math.pi,0.0,True)
    elif choice == 3: #S Gate
        unitary_gate = (-math.pi/2,math.pi,math.pi,math.pi,False)
    elif choice == 4: #T Gate
        unitary_gate = (-math.pi/4,math.pi/2,2*math.pi,0.0,False)
    delta,alpha,theta,beta,value = unitary_gate
    return delta,alpha,theta,beta,value

#code which takes an angle and alters the gate
def random_altered_unitary_gate(delta,alpha,theta,beta,value):
    if delta == 0.0 and alpha == 0.0 and theta == math.pi and value == True:
        angles = ['delta','alpha','beta']
    else:
        angles = ['delta','alpha','theta','beta']
    altered_variable = choice(angles)
    if altered_variable == 'delta':
        delta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'alpha':
        alpha = uniform(0.0,2.0*math.pi)
    if altered_variable == 'theta':
        theta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'beta':
        beta = uniform(0.0,2.0*math.pi)
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate
   
#gives both an original and altered unitary gate
#can be commented to return oritinal gate, corruspondig altered gate(onle one thing different from original), or both
def unitary_gate(choice):
    delta,alpha,theta,beta,value = random_angles()
    original = random_unitary_gate(delta,alpha,theta,beta,value)
    matrix=original
    if choice:
        altered = random_altered_unitary_gate(delta,alpha,theta,beta,value)
        matrix=altered
    return (matrix,[delta,alpha,theta,beta,value])

"""
CNOT & CZ functions
"""
#alter a CNOT
def rot(qubits,choice):
    a=randint(0,qubits-2)
    b=randint(a+1,qubits-1)
    k=qt.cnot(qubits,a,b)
    k=k.full()
    if choice:
        k=np.random.permutation(k)
        k=qt.Qobj(k)
        #k=k*k.dag()
        k=tensor_fix(k)
    cn_final=qt.Qobj(k)
    return cn_final

def conv_cz():
    gate=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,-1]])
    gate=qt.Qobj(gate)
    return gate

"""
Classification and Alteration Protocols
"""
def categorize(circuit):
    cat=[]
    it_not=1
    it_h=1
    it_rand=1
    it_id=0
    composition=[]
    
    
    for i in range(len(circuit)):
        if i % 2 != 0:
            continue
        if circuit[i+1] in cat:
            if "Hadamard" in circuit[i+1]:
                it_h+=1
                t=str(it_h)
                t="Hadamard" + t
                cat.append(t)
            elif "CNOT" in circuit[i+1]:
                it_not+=1
                t=str(it_not)
                t="CNOT" + t
                cat.append(t)
            elif "Random Unitary" in circuit[i+1]:
                it_rand+=1
                t=str(it_rand)
                t="Random Unitary" + t
                cat.append(t)
        elif "Identity" in circuit[i+1]:
            it_id +=1
            t=str(it_id)
            t="Measurement" + t
            cat.append(t)
        else:
            cat.append(circuit[i+1])
        i+=2
    return(cat)
    
def rabbit(qubits,choice,start):
    basic_0ket=qt.Qobj([[1],[0]])
    basic_1ket=qt.Qobj([[0],[1]])
    if start ==1:
        temp=basic_1ket
    else:
        temp=basic_0ket
    r=1
    while r < qubits:
        if r in choice:
            temp=qt.tensor(temp,basic_1ket)
            temp=tensor_fix(temp)
        else:
            temp=qt.tensor(temp,basic_0ket)
            temp=tensor_fix(temp)
        r+=1
    return temp
    
def gen_basis_vectors(n,dims,choice):
    vectors=[]
    basic_states=[]
    bits=int(math.log(n,2))
    #for i in range(n):
        #state=qt.basis(n,i)
        #fock_states.append(state)
    q=rabbit(bits,[],0)
    basic_states.append(q)
    q=rabbit(bits,[1],0)
    basic_states.append(q)
    q=rabbit(bits,[n-1],0)
    basic_states.append(q)
    indexvec=[]
    for i in range(bits):
        indexvec.append(i)
    q=rabbit(bits,indexvec,1)
    basic_states.append(q)
    test_opt1=[basic_states[0],basic_states[1],basic_states[2],basic_states[3]]
    test_opt2=[basic_states[0],basic_states[1],basic_states[2],basic_states[3],basic_states[-1]+basic_states[1]]
    #test_opt1=[fock_states[0],fock_states[1],fock_states[2],fock_states[3],fock_states[-1]+fock_states[1]]
    #test_opt2=[fock_states[0],fock_states[0]+fock_states[1],fock_states[-1]+fock_states[0]+fock_states[1],fock_states[-2]+fock_states[1]+fock_states[0]+fock_states[-1],fock_states[-2]+fock_states[1]+fock_states[0]+fock_states[-1]]
    if choice == 1: #Basis states
        vectors = basic_states
        q=qt.Qobj(np.ones(n))
        q=q.unit()
        vectors.append(q)
        #vectors.append(state) #there is no two for now
    elif choice ==2:
        vectors=basic_states
    elif choice == 3: #Hadamard option 1
        """
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in test_opt1:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
        """
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 4: #QFT option 1
        quft=tensor_fix(qft.qft(bits))
        #for state in test_opt1:
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard option 2
        h=tensor_fix(qt.hadamard_transform(bits))
        #for state in test_opt2:
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #QFT option 2
        quft=tensor_fix(qft.qft(bits))
        #for state in test_opt2:
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors



"""
"""
def get_ideal(circuit,vectors,qubits,d): 
    probabilities=[]
    n=2**qubits
    """
    references=[]
    for chi in range(d):
        compare=gen_basis_vectors(n,n,4)
        references.append(compare[chi])
    """
    references=vectors
    temparray=[]
    i=0
    for ref in references:
        state=vectors[i]
        final=basic_b(state,circuit)
        prob=dis(final,ref)
        temparray.append(prob)
        i+=1
    probabilities.append(temparray)
    return probabilities
"""
def within_tolerance(tolerance,probvector,idealvector):
    truth=1
    scale=1-tolerance
    for i in range(len(idealvector)):
        test=probvector[i]
        wiggle=scale*idealvector[i]
        minm=idealvector[i]-wiggle
        maxm=idealvector[i]+wiggle
        if test <= maxm and test >= minm:
            truth=truth*1
        else:
            truth=truth*0
    return truth
"""
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
    d=4
    qubits=2
    n=2**qubits
    th=uniform(0,2*np.pi)
    rotate=tensor_fix(qt.tensor(id2,qt.ry(th)))
    rotate=tensor_fix(qt.tensor(rotate,id2))
    """
    state_creator=[rotate*hadamaker(qubits,[1]),qt.cnot(qubits,1,2),qt.cnot(qubits,0,1),hadamaker(qubits,[0]),qt.cnot(qubits,1,2),conv_cz()]
    state_creator_tags=["Hadamard","CNOT","CNOT2","Hadamard2","CNOT3","Control Z"]
    circuit=[]

    for i in range(len(state_creator)):
        circuit.append(state_creator[i])
        circuit.append(state_creator_tags[i])
    alt=[]
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))
    """
    ghzcirc=[hadamaker(qubits,[0]),qt.cnot(qubits,0,1)]
    ghz_tags=["Hadamard","CNOT"]
    circuit=[]
    for i in range(len(ghzcirc)):
        circuit.append(ghzcirc[i])
        circuit.append(ghz_tags[i])
    #cat=categorize(circuit)
    alt=[]
    #angles=[]
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))
    choice = [2,4,5,6]
    vector_name = ['new','QFT','Hadamard 2','Fourier State']
    #ideals=[]
    tolerance=input("Give a tolerance: ")
    tolerance=float(tolerance)
    vectors=gen_basis_vectors(n,n,5)
    ideal=get_ideal(alt,vectors,qubits,d)
    print(vector_name[i])
    print(ideal[0])
    ideal=[0.7803300858899107, 0.7803300858899107, 0.7803300858899107, 0.7803300858899107]
    #ideals.append(ideal[0])

    
    path=input("Give csv file location, remembering to use forward slashes: ")
    KNN(path,0.8,5,tolerance,ideal)
    """
    tolerance=0.9
    for i in range(10):
        multi=uniform(0,1)
        ideal2=ideal[0]
        probvector=[]
        for x in range(len(ideal2)):
            probvector.append(multi*ideal2[x])
        truth=within_tolerance(tolerance,probvector,ideal[0])
        print(multi)
        print(truth)
    """
    return 0
    
main()