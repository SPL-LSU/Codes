#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roy Pace
#7.13.2020
#An attempt to implement Shor's Factoring Algorithm

#https://arxiv.org/pdf/quant-ph/0205095.pdf


# In[17]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import math as math
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


def QuantumFourierTransform(qc, numqubits, startqubit):
    #applies the hadamard 
    for i in range(startqubit,numqubits+startqubit):
        qc.h(q[i])
    #applies the appropriate rotation gates
        for j in range(i+1,numqubits+startqubit):
                qc.cu1(math.pi/(2**(j)),q[j],q[i])

    #Swap gate is not needed            
    #for i in range(0,int(numqubits/2)):
        #qc.swap(i,numbqubits-i-1)


# In[46]:


def InverseQFT(qc, numqubits, startqubit):
    for i in range((numqubits+startqubit)//2):
        qc.swap(i, numqubits+startqubit-i-1)
    for j in range(numqubits+startqubit):
        for i in range(j):
            qc.cu1(-np.pi/float(2**(j-i)),i,j)
        qc.h(j)


# In[45]:


def BigHadamard(qc, numqubits, startqubit):
    for i in range(numqubits+startqubit):
        qc.h(i)


# In[ ]:


def c_amod15(qc, a, power, startqubit):
    if a not in [2,7,8,11,13]:
        raise ValueError("a Must be 2,7,8,11,13")
    for iteration in range(power):
        if a in [2,13]:
            qc.swap(0+startqubit,1+startqubit)
            qc.swap(1+startqubit,2+startqubit)
            qc.swap(2+startqubit,3+startqubit)
        if a in [7,8]:
            qc.swap(2+startqubit,3+startqubit)
            qc.swap(1+startqubit,2+startqubit)
            qc.swap(0+startqubit,1+startqubit)
        if a == 11:
            qc.swap(1+startqubit,3+startqubit)
            qc.swap(0+startqubit,2+startqubit)
        if a in [7,11,13]:
            for q in range(4):
                qc.x(q+startqubit)
   # U = U.to_gate()
   # c_U = U.control()


# In[ ]:


def Ua(qc, numqubits, startqubit):
    control(1)


# In[59]:


#Create Circuit
n=4
m = 3*n #number of quibits in circuit
q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)

#Order Finding Circuit

#Setup
BigHadamard(qc, 2*n , 0)
qc.x(m-1)

#Ideally the following unitary would be generalized for n quibits
#but I'm still working on it at the moment
#AKA modular exponentiation
for q in range(2*n):
    (c_amod15(qc, a, 2**q, 2*n),[q] + [i+2*n for i in range(n)])


InverseQFT(qc,2*n,0)

qc.measure(q,c)
qc.draw()


# In[36]:


from qiskit.aqua.algorithms import Shor
a, N = 2, 3
shor = Shor(N, a)
circuit = shor.construct_circuit()
#print(circuit.draw())  
#circuit.draw(output='mpl')


# In[ ]:


#U2, U1, U3


# In[49]:


#Conditional Phase Shift
def PhaseShift(qc, theta, qubit, ctrl):
    qc.u1(theta, qubit).control(1)
    qc.x(qubit).control(1)
    qc.u1(theta, qubit).control(1)
    qc.x(qubit).control(1)
#def ConditionalPhaseShift(qc, theta, qubit):
    #PhaseShift(qc, theta,qubit).control(num_ctrl_qubits=1, ctrl_state)# not sure how to make control act on a specific qubit?


# In[57]:


#Part 2: Fidelity Data


# In[ ]:




