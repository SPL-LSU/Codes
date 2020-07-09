
import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
import time
import csv
import scipy.linalg as la

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
            k=tensor_fix(k)
            state=qt.Qobj(state)
            state=k(state)
    return state
    
#Fidelity distance metric
def dis(state1,state2):
    (n,m)=state1.shape
    if state1.type != 'ket' or state2.type != 'ket':
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
    return gate

"""
Hadamard generation gates
"""

#messed up hadamard
def multi_qubit_hadamard(regular_hadamard_gate):
    theta = uniform(0.0,math.pi*2.0)
    (n,n) = regular_hadamard_gate.shape
    N = np.int(((np.log(n))/(np.log(2))))
    phase=qt.qip.operations.globalphase(theta,N)
    phase=tensor_fix(phase)
    reg=tensor_fix(regular_hadamard_gate)
    multi_qubit_hadamard = phase*reg
    multi_qubit_hadamard = tensor_fix(multi_qubit_hadamard)    
    return multi_qubit_hadamard

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
            if count == 200: #no infinite loops
                break
        #initialize test unitary
        if seed == 0:
            u1=qt.qip.operations.hadamard_transform(1)
        else:
            u1=id2
        #create test unitary
        while u1.shape != (n,n):
            if i ==seed: #set a hadamard on specified qubit
                u1=qt.tensor(u1,qt.qip.operations.hadamard_transform(1))
                u1=tensor_fix(u1)
            else:
                u1=qt.tensor(u1,id2)
                u1=tensor_fix(u1)
            i+=1
        check=u1*hada
        if check.full()[0][0] > mag: #if there's a hadamard on that qubit, true
            ongoing = False
        elif count == 200:
            print("oops")
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
        gate=qt.qip.operations.phasegate(theta)
    elif phaser == 1:
        gate=qt.qip.operations.rz(theta)
    elif phaser==2:
        gate = qt.qip.operations.ry(theta)
    else:
        gate=qt.qip.operations.globalphase(theta)
    #alter gate
    if seed == 0:
        u1=gate
    else:
        u1=id2
    i=1
    while u1.shape != (n,n):
        if i ==seed: #set a alteration on specified qubit
            u1=qt.tensor(u1,gate)
            u1=tensor_fix(u1)
        else:
            u1=qt.tensor(u1,id2)
            u1=tensor_fix(u1)
        i+=1
    final_gate=u1*hada
    return final_gate

def h_reassign(hada):

    seed = randint(0,1)
    if seed == 0:  # alter whole hadamard
        alt_had=multi_qubit_hadamard(hada)
    if seed == 1:  # alter specific hadamard
        (h,seed)=hadamard_preprocessing(hada)
        alt_had=alter_hadamard(hada,seed)

    return alt_had


def hadamaker(qubits,affected):
    # affect is list of affected qubits, 0 indexed
    # makes an n qubit hadamard, affecting a subset of qubits
    array=[]
    i=0
    while len(array) < qubits:
        if i in affected:
            gate=qt.qip.operations.hadamard_transform()
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

def random_unitary_gate(delta,alpha,theta,beta,value):
    # code which give an original unitary gate
    gate = qt.Qobj(qt.qip.operations.phasegate(delta)*qt.qip.operations.rz(alpha)*qt.qip.operations.ry(theta)*qt.qip.operations.rz(beta))
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


def random_altered_unitary_gate(delta, alpha, theta, beta, value):
    # code which takes an angle and alters the gate
    if delta == 0.0 and alpha == 0.0 and theta == math.pi and value == True:
        angles = ['delta','alpha','beta']
    else:
        angles = ['delta','alpha','theta','beta']

    altered_variable = choice(angles)
    if altered_variable == 'delta':
        delta = uniform(math.pi, 2.0*math.pi)
    if altered_variable == 'alpha':
        alpha = uniform(math.pi, 2.0*math.pi)
    if altered_variable == 'theta':
        theta = uniform(math.pi, 2.0*math.pi)
    if altered_variable == 'beta':
        beta = uniform(math.pi, 2.0*math.pi)
    gate = qt.Qobj(qt.qip.operations.phasegate(delta)*qt.qip.operations.rz(alpha)*qt.qip.operations.ry(theta)*qt.qip.operations.rz(beta))

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

def rot(qubits,choice):

    a=randint(0,qubits-2)
    b=randint(a+1,qubits-1)
    k=qt.qip.operations.cnot(qubits,a,b)
    k=k.full()
    if choice:
        k=np.random.permutation(k)
        k=qt.Qobj(k)
        k=tensor_fix(k)
    cn_final=qt.Qobj(k)

    return cn_final


def conv_cz(alt):

    gate=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                   [0,0,0,0,1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,-1]])
    gate=qt.Qobj(gate)
    if alt == True:
        gate = np.random.permutation(gate)

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


def normalize_state(state):
    state = np.array(state)
    state = state / sl.norm(state)

    return state


def gen_basis_vectors(n,dims,choice):

    vectors=[]
    bits=int(math.log(n,2))
    basic_states = [np.array([0,1,0,0,0,0,0,0]).T, np.array([0,0,0,1,0,0,0,0]).T, np.array([0,0,0,0,0,1,0,0]).T, np.array([0,0,0,0,0,0,0,1]).T]
    basic_states = [qt.Qobj(x) for x in basic_states]

    if choice ==2: #new
        vectors=basic_states
    elif choice == 4: #QFT
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard
        h=tensor_fix(qt.qip.operations.hadamard_transform(bits))
        for state in basic_states:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #Fourier State
        quft=tensor_fix(qft.qft(bits))
        for state in basic_states:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors


def colin_mochrie(circuit, vectors, pop, cat, qubits, d, path, choice, regions):
    probabilities = []
    n = 2**qubits
    index = 0
    for j in range(pop):
        references = []
        for chi in range(d):
            compare = gen_basis_vectors(n,n,choice)
            references.append(compare[chi])

        for i in range(len(circuit)):
            gate_holder = circuit[i]
            name = cat[i]
            if "Hadamard" in name:
                alt_gate=h_reassign(gate_holder)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(h_reassign(circuit[i]),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break

            elif "CNOT" in name:
                alt_gate=rot(qubits,True)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(rot(qubits,True),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break

            elif "Control" in name: # check that this cz gate was not generating variation before you do this.
                alt_gate = conv_cz(True)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(rot(qubits,True),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break

            temparray = []
            t = 0
            for ref in references:
                state = vectors[t]
                final = basic_b(state,circuit)
                #print('_________')
                #print(state)
                #print(ref)
                #print('_________')
                prob = dis(final,ref)
                temparray.append(prob)
                t+=1
            temparray.append(name) #regions[i]
            probabilities.append(temparray)
            print(temparray)

            with open(path,'a',newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(temparray)
            csvFile.close()

            count=0
            while circuit[i] != gate_holder:
                circuit[i] = gate_holder
                count+=1
                if count == 10:
                    break

    return probabilities
                
"""
Main Party Time!
"""

def main():
    
    pop = 30
    d = 4
    qubits = 3
    n = 2 ** qubits

    csvpath = ["TeleportTrainingData500new.csv","TeleportTrainingData500QFT.csv",
             "TeleportTrainingData500Had.csv","TeleportTrainingData500QFT2.csv"]
    state_creator = [hadamaker(qubits,[1]), qt.qip.operations.cnot(qubits,1,2), qt.qip.operations.cnot(qubits,0,1),
                   hadamaker(qubits,[0]), qt.qip.operations.cnot(qubits,1,2), conv_cz(False)]
    regions = ["Region1", "Region1", "Region1", "Region2", "Region2", "Region2"]
    state_creator_tags = ["Hadamard","CNOT","CNOT2","Hadamard2","CNOT3","Control Z"]
    circuit=[]

    for i in range(len(state_creator)):
        circuit.append(state_creator[i])
        circuit.append(state_creator_tags[i])
    cat=categorize(circuit)
    alt=[]

    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))

    choice = [2,4,5,6]
    index1 = 0

    for x in choice:
        choice = x
        path = csvpath[index1]
        vectors = gen_basis_vectors(n,n,choice)
        index1 += 1
        colin_mochrie(alt, vectors, pop, cat, qubits, d, path, choice, regions)

    return 0

start = time.time()
main()
print(time.time()-start)
