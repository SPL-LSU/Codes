# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:15:23 2020

@author: marga
"""

import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
import time
import csv

def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

id2=qt.Qobj(np.identity(2))

#Finds the euclidean distance bewteen two vectors of length 'length;
def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)

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

def paulix(qubits,pos):
    if pos == 0:
        gate=qt.Qobj(np.array([[0,1],[1,0]]))
    else:
        gate=id2
    for i in range(1,qubits):
        if i == pos:
            temp=qt.Qobj(np.array([[0,1],[1,0]]))
        else:
            temp=id2
        gate=tensor_fix(qt.tensor(gate,temp))
    gate=qt.Qobj(gate)
    return gate

def alter(gate_holder):
    (ugate,angles)=unitary_gate(1)
    alt_gate=ugate*gate_holder
    alt_gate=qt.Qobj(alt_gate)
    return alt_gate

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
    
def get_ideal(circuit,basis,vectors,qubits,d): 
    probabilities=[]
    n=2**qubits
    references=[]
    for chi in basis:
        references.append(chi)
    """
    for chi in range(d):
        compare=gen_basis_vectors(n,n,4)
        references.append(compare[chi])
    """
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

def get_single_gate_data():
    inputs=[]
    h=tensor_fix(qt.hadamard_transform())
    zero=tensor_fix(qt.fock(2,0))
    one=tensor_fix(qt.fock(2,1))
    inputs.append(h*zero)
    inputs.append(h*one)
    inputs.append(h*zero)
    inputs.append(h*one)
    gate=paulix(1,0)
    ideal=get_ideal([gate],inputs,inputs,1,4)
    tag="X"
    pop=input("how may data points? ")
    pop=int(pop)
    path="SingleDoubleHada2000.csv"
    probabilities=[]
    for i in range(pop):
        alt_gate=alter(gate)
        temparray=[]
        for state in inputs:
            final=basic_b(state,[alt_gate])
            prob=dis(final,state)
            temparray.append(prob)
        if within_tolerance(0.78,temparray,ideal[0]):
            temparray.append("tolerance")
        else:
            temparray.append(tag)
        probabilities.append(temparray)
        with open(path,'a',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(temparray)
        csvFile.close
    return probabilities
    
get_single_gate_data()