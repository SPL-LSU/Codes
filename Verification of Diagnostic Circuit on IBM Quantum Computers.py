# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:34:36 2020

@author: aliza siddiqui

This program creates the diagnostic circuit shown in the following paper:

("Finding Broken Gates in Quantum Circuits---Exploiting Hybrid Machine Learning"
https://arxiv.org/abs/2001.10939)
"""
from qiskit import *
import math as math

#Sets U unitary to multi-qubit hadamard
def setMulti_Qubit_Had(qc):
    for i in range(4):
        qc.h(i)
        qc.barrier(0,1,2,3)

#Sets V unitary to Quantum Fourier Transform        
def set_QFT(qc):
    #applies the hadamard 
    for i in range(4,8):
        qc.h(i)
        #applies the appropriate rotation gates
        for j in range(i+1,8):
                qc.cu1(math.pi/(2**(j)),j,i)

    qc.barrier(4,5,6,7)

#Creating Repeater Circuit        
def createRepeater(qc):
    qc.h(1)
    qc.h(3)
    qc.cx(1,0)
    qc.cx(3,2)
    for i in range(4):
        qc.h(i)
    qc.cx(2,1)
    qc.h(1)
    qc.h(2)
    qc.cx(2,1)
    qc.h(1)
    qc.h(2)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(0)
    qc.cx(2,0)
    qc.h(2)
    qc.h(0)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)
    qc.h(2)
    qc.h(3)
    qc.cx(3,2)

    qc.barrier(0,1,2,3)
    qc.barrier(4,5,6,7)

#Creates Swap Test portion of circuit    
def swapTest(qc, c):
    qc.h(8)

    qc.cswap(8, 4, 0)
    qc.cswap(8, 5, 1)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)

    qc.h(8)

    qc.measure(8, c)



def main():
    n = 9 #number of qubits in diagnostic circuit
    q = QuantumRegister(n)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    setMulti_Qubit_Had(qc)
    createRepeater(qc)
    set_QFT(qc)
    swapTest(qc, c)
    

   # qc.draw(output = "mpl")
    print(qc)
    
main()







