#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roy Pace
#7.15.2020
#An attempt to implement the Deutsch-Josza Algorithm in Qiskit


# In[1]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
import matplotlib.pyplot as plt
import math as math
import numpy as np
from qiskit.visualization import plot_histogram
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def BigHadamard(qc, numqubits, startqubit):
    for i in range(numqubits+startqubit):
        qc.h(i)
    qc.barrier()


# In[3]:


def ConstOracle(qc, qubit):
    output = np.random.randint(2)
    if output == 1:
        qc.x(qubit)


# In[4]:


def BalancedOracle(qc, b_str, num_qubit):
    for i in range(len(b_str)):
        if b_str[i] == "1":
            qc.x(i)
    for i in range(num_qubit):
        qc.cx(i, num_qubit)
    for i in range(len(b_str)):
        if b_str[i] == "1":
            qc.x(i)


# In[17]:


#Create Circuit
m = 4 #number of quibits in circuit
q = QuantumRegister(m)
c = ClassicalRegister(m)
qc = QuantumCircuit(q,c)
qc1 = QuantumCircuit(QuantumRegister(1), ClassicalRegister(1))

#Order Finding Circuit

#Setup
BigHadamard(qc, m , 0)
qc1.x(0)
qc1.h(0)

#Implement oracle

qc += qc1

qc.barrier()

BalancedOracle(qc, "101", m)
#ConstOracle(qc, m)

qc.barrier()

BigHadamard(qc, m, 0)


qc.measure(q,c)
qc.draw()


# In[18]:


backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(qc, backend=backend, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[8]:


from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')

shots = 1024

provider = IBMQ.get_provider(group='open')

backend = provider.get_backend('ibmq_16_melbourne')
i=1
sum = []
counts = []
avg = 0
for i in range(1,13):
    
    exp_job = execute(qc, backend)
    job_monitor(exp_job)
    exp_counts = exp_job.result().get_counts()
    counts.append(exp_counts)
    sum.append(exp_counts["0 1111"]/shots)
    i+=1
#avg = sum/(i-1)
print(sum)
#plot_histogram(exp_counts)


# In[22]:


from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')

shots = 1024

provider = IBMQ.get_provider(group='open')

backend = provider.get_backend('ibmq_16_melbourne')
i=1
sum = []
counts = []
avg = 0
for i in range(1,13):
    
    exp_job = execute(qc, backend)
    job_monitor(exp_job)
    exp_counts = exp_job.result().get_counts()
    counts.append(exp_counts)
    sum.append(exp_counts["0 1111"]/shots)
    i+=1
#avg = sum/(i-1)
print(sum)
print(exp_counts)
#plot_histogram(exp_counts)


# In[31]:


avg=0
i=0
for i in range(0,12):
    avg+=sum[i]
    print(i)
avg=avg/(i+1)


# In[32]:


avg


# In[ ]:


#from qiskit import *
#from qiskit.visualization import plot_histogram
#from qiskit.tools.monitor import job_monitor
#IBMQ.load_account()
#provider = IBMQ.get_provider(group='open')


#provider = IBMQ.get_provider(group='open')

#backend = provider.get_backend('ibmq_essex')
#exp_job = execute(qc, backend)
#job_monitor(exp_job)
#exp_counts = exp_job.result().get_counts()
#plot_histogram(exp_counts)


# In[33]:


#Fidelity Calculations
counts


# In[ ]:


exp_counts["0 0000"]/shots


# In[ ]:


answer["0 0000"]/shots


# In[26]:


from qiskit.quantum_info import state_fidelity
state_fidelity(answer,exp_counts)


# In[27]:


state_fidelity()


# In[11]:


import pickle

with open('DJ_outfile.txt', 'wb') as fp:
    pickle.dump(sum, fp)


# In[ ]:




