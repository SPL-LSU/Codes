#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roy Pace
#7.13.2020
#An attempt to implement Shor's Factoring Algorithm

#https://arxiv.org/pdf/quant-ph/0205095.pdf


# In[12]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import math as math
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


def quantumFourierTransform(qc, numqubits, startqubit):
    #applies the hadamard 
    for i in range(startqubit,numqubits+startqubit):
        qc.h(q[i])
    #applies the appropriate rotation gates
        for j in range(i+1,numqubits+startqubit):
                qc.cu1(math.pi/(2**(j)),q[j],q[i])

    #Swap gate is not needed            
    #for i in range(0,int(numqubits/2)):
        #qc.swap(i,numbqubits-i-1)


# In[89]:


def inverseQFT(qc, numqubits, startqubit):
    for i in range((numqubits+startqubit)//2):
        qc.swap(i, numqubits+startqubit-i-1)
    for j in range(numqubits+startqubit):
        for i in range(j):
            qc.cu1(-np.pi/float(2**(j-i)),i,j)
        qc.h(j)


# In[97]:


def bigHadamard(qc, numqubits, startqubit):
    for i in range(numqubits+startqubit):
        qc.h(i)


# In[ ]:


#(a*x) mod N
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


# In[63]:


#actually just restart, using the circuit in the paper
#Create Circuit
m = 5 #number of quibits in circuit
q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)

#Order Finding Circuit

#Setup
BigHadamard(qc, 3 , 0)
qc.cx(2,3)
qc.cx(2,4)
QuantumFourierTransform(qc, 3, 0)



qc.measure(q,c)
qc.draw()


# In[64]:


#Execute the Circuit
from qiskit.circuit import Gate
from qiskit import execute, Aer

shots = 20000
job = execute(qc, Aer.get_backend('qasm_simulator'), shots=shots)
counts = [job.result().get_counts(i) for i in range(len(job.result().results))]
from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[ ]:


def Unitary(qc, numqubits, startqubit):
    


# In[4]:


import numpy as np
#repeated squaring algorithm to calculate a^2^j (mod N)
def modPrep(a, j, N):
    for i in range(j):
        a = np.mod(a**2, N)
    return a


# In[9]:


def Ph(qc, theta, qubit):
    qc.u1(theta, qubit)
    qc.x(qubit)
    qc.u1(theta,qubit)
    qc.x(qubit)
#Creates
#[[e^(i*theta),0],
#[0,e^(i*theta)]]


# In[ ]:


#conditional rotation, j = k in papers
#qc.cu1(math.pi/(2**(j)),q[j],q[i])


# In[43]:


#first gate, 1 to n
#second gate, 1 to n-1
#cu1, control , target

for i in range(0,m):
    #applies the appropriate rotation gates
    for j in range(i+1,m):
            qc.cu1(math.pi/(2**(j)),q[j],q[i])


# In[82]:


def additionTransform(qc, num_gates,start_qubit):
    k=0
    for i in range(start_qubit,num_gates+start_qubit):
        for j in range(0,start_qubit-k):
            qc.cu1(math.pi/2**(j), q[j+k], q[i])
        qc.barrier()
        k+=1


# In[91]:


#Create Circuit
n=4
m = 8 #number of quibits in circuit
N = 5
q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)

#Order Finding Circuit

additionTransform(qc,int(m/2),int(m/2))
additionTransform(qc, N, int(m/2)) #N addition is probably wrong
inverseQFT(qc,int(m/2),int(m/2))
qc.cx(m-2,m-1)
quantumFourierTransform(qc,int(m/2),int(m/2))
additionTransform(qc, N, int(m/2))
additionTransform(qc,int(m/2),int(m/2))
inverseQFT(qc,int(m/2),int(m/2))
qc.x(m-2)
qc.cx(m-2,m-1)
qc.x(m-2)
quantumFourierTransform(qc,int(m/2),int(m/2))
additionTransform(qc,int(m/2),int(m/2))


qc.measure(q,c)
qc.draw()


# In[92]:


def modularAdderGate(qc,m,N):
    additionTransform(qc,int(m/2),int(m/2))
    additionTransform(qc, N, int(m/2)) #N addition is probably wrong
    inverseQFT(qc,int(m/2),int(m/2))
    qc.cx(m-2,m-1)
    quantumFourierTransform(qc,int(m/2),int(m/2))
    additionTransform(qc, N, int(m/2))
    additionTransform(qc,int(m/2),int(m/2))
    inverseQFT(qc,int(m/2),int(m/2))
    qc.x(m-2)
    qc.cx(m-2,m-1)
    qc.x(m-2)
    quantumFourierTransform(qc,int(m/2),int(m/2))
    additionTransform(qc,int(m/2),int(m/2))


# In[94]:


def cmultGate(qc,m,N):
    quantumFourierTransform(qc,int(m/2),int(m/2))
    for i in range(0, n-1):
        modularAdderGate(qc,m,N) #need some sort of 2**n * a??
    inverseQFT(qc,int(m/2),int(m/2))


# In[152]:


def cUGate(qc,m,N,k,n):
    cmultGate(qc,m, N)
    for i in range(n, 2*n):
        p=0
        #k goes from 0 to 2n
        #i goes from 0 to int(m/2)
        #qc.cswap(k,i, i+int(m/2))
        
    cmultGate(qc,m,N) #supposed to be a^-1??


# In[153]:


#Create Circuit
n=4
m = 8 #number of qubits in circuit
N = 5

q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)

#Order Finding Circuit

bigHadamard(qc, 2*n, 0)
for k in range(0, 2*n):
    cUGate(qc,n,N,k,n)
inverseQFT(qc, 2*n, 0)


qc.measure(q,c)
qc.draw()


# In[139]:


m = 3

q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)

qc.cswap(1,2,0)

qc.measure(q,c)
qc.draw()


# In[ ]:


for i in range(0,m):
    #qc.h(q[i])
    #applies the appropriate rotation gates
    for j in range(i+1,m):
            qc.cu1(math.pi/(2**(j)),q[j],q[i])

