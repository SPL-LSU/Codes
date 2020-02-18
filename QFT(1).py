#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roy Pace
#1/27/2020
#This code produces a Quantum Fourier Transform for m number of qubits


# In[2]:


#Papers Looked At:
#https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html#example1
#https://arxiv.org/pdf/1903.04359.pdf
#https://quantum-computing.ibm.com/support
#https://qiskit.org/documentation/api/qiskit.extensions.standard.Cu1Gate.html?highlight=cu1
#http://pages.cs.wisc.edu/~dieter/Papers/vangael-thesis.pdf
#https://arxiv.org/pdf/1804.03719.pdf


# In[3]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import math as math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Create the Circuit
m=1 #number of quibits, must be greater than 0
q=QuantumRegister(m)
c=ClassicalRegister(m)
qc= QuantumCircuit(q,c)


# In[5]:


#applies the hadamard 
for i in range(0,m):
    qc.h(q[i])
    #applies the appropriate rotation gates
    for j in range(i+1,m):
            qc.cu1(math.pi/(2**(j)),q[j],q[i])

#Swap gate is not needed            
#for i in range(0,int(m/2)):
    #qc.swap(i,m-i-1)
qc.measure(q,c)
qc.draw()


# In[6]:


#Execute the Circuit
from qiskit.circuit import Gate
from qiskit import execute, Aer

shots = 20000
job = execute(qc, Aer.get_backend('qasm_simulator'), shots=shots)
counts = [job.result().get_counts(i) for i in range(len(job.result().results))]
from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[ ]:




