"""
Created on Tue May 19 12:54:25 2020
Redoing Hadamard Alteration for less functions and errors.

Coding comments:
Needs to be tested by should function the same way.
Not pretty, but elimates decomposition of unitaries and generation of unused combinations.

Physics explanation:
Uses the fact that the magnitude of nonzero elements in a hadamard tell how many hadamards are applied
as well as the fact that two hadamards in succession cancel to an identity, so you can identify a 
hadamard by applying a second hadamard and observing if the magnitude goes down.

@author: Margarite L. LaBorde
"""

import qutip as qt
import numpy as np
from random import randint, uniform

"""necessary functions"""
def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

id2=qt.Qobj(np.identity(2))
"""begin simplified code"""

#redifine h_reassign
def h_reassign(hada):
    seed = randint(0,1)
    if seed == 0: #alter whole hadamard
        alt_had=multi_qubit_hadamard(hada)
    if seed == 1: #alter specific hadamard
        alt_had=alter_hadamard(hada)
    return alt_had
    
#Looks awful, I promise it isn't. while loops are mostly security
#Just figures out whether there's a hadamard present we can alter
def hadamard_preprocessing(hada):
    storage=hada.full()
    (n,n)=hada.shape
    q=np.log(n)/np.log(2) #number of qubits
    seed=randint(0,q-1)
    forbidden=[] #a vector to hold forbidden seeds
    mag=storage[0][0] #magnitude of the elements in the hadamard
    ongoing=True
    while ongoing:
        i=1
        count=0
        while seed in forbidden: #make sure the seed isn't forbidden
            seed=randint(0,q-1)
            count+=1
            if count == 20: #no infinite loops
                break
        #initialize test unitary
        if seed == 0:
            u1=qt.hadamard_tranform(1)
        else:
            u1=id2
        #create test unitary
        while u1.shape != (n,n):
            if i ==seed: #set a hadamard on specified qubit
                u1=qt.tensor(u1,qt.hadamard_transform(1))
                u1=tensor_fix(u1)
            else:
                u1=qt.tensor(u1,id2)
                u1=tensor_fix(u1)
            i+=1
        check=u1*hada
        if check.full()[0][0] < mag: #if there's a hadamard on that qubit, true
            ongoing = False
        elif count == 20:
            break
        else: #no hadamard on that seed qubit
            forbidden.append(seed)
    return hada,seed

#can feed as input the preprocessing step
def alter_hadamard(hada,seed):
    (n,n)=hada.shape
    theta = uniform(0.0,math.pi*2.0)
    
    #pick a rotation any rotation
    phaser=randint(0,3)
    if phaser ==0:
        gate=qt.phasegate(theta)
    elif phaser == 1:
        gate=qt.rz(theta)
    elif phaser==2:
        gate = qt.ry(theta)
    else:
        gate=qt.gloablphase(theta)
        
    #alter gate
    if seed == 0:
        u1=gate
    else:
        u1=id2
    i=1
    while u1.shape != (n,n):
        if i ==seed: #set a alteration on specified qubit
            u1=qt.tensor(u1,qt.globalphase(theta,1))
            u1=tensor_fix(u1)
        else:
            u1=qt.tensor(u1,id2)
            u1=tensor_fix(u1)
        i+=1
    final_gate=u1*hada
    return final_gate
