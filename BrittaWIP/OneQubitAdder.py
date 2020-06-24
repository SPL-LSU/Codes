""""
Generate training data for a one-qubit adder circuit
Source: "Approaching Quantum Computing" by Dan Marinescu

BNM Jun 23 '20

"""

import numpy as np
import qutip as qt
import math
from random import randint, uniform, choice,random
import time
import csv

#===============================================================================#

def rabbit(temp_gate, qubits, index):
    """
    Tensors the gate to correct size
    """
    temp_gate = temp_gate.full()

    if index == 0:
        temp = temp_gate
    else:
        temp = np.eye(2)
    for i in range(qubits):
        if i == 0:
            continue
        if i == index:
            temp = np.kron(temp, temp_gate)
        else:
            temp = np.kron(temp, np.eye(2))

    return qt.Qobj(temp)


#===============================================================================#

def basic_b(state, array):
    """
    Applies an array of gates in a circuit and applies them to a state
    """
    state = state.full()

    for i in range(len(array)):
        operator = array[i].full()
        state = operator.dot(state)

    return state

#===============================================================================#

def dis(state1, state2):
    """
    Takes the "distance" (Fidelity metric) between the input & reference state
    :return:
    """
    state1 = qt.Qobj(state1)
    state2 = qt.Qobj(state2)
    (n,m) = state1.shape

    fid = qt.fidelity(state1,state2)
    p_0 = 1/n + (n-1)*fid/n

    return p_0

#===============================================================================#

def random_unitary_gate(delta,alpha,theta,beta):

    gate = qt.Qobj(qt.qip.operations.phasegate(delta)*qt.qip.operations.rz(alpha)*\
                   qt.qip.operations.ry(theta)*qt.qip.operations.rz(beta))

    return gate

#===============================================================================#

def tgate():

    gate = qt.Qobj(qt.qip.operations.phasegate(-math.pi/4)*qt.qip.operations.rz(math.pi/2)\
                   *qt.qip.operations.ry(2*math.pi)*qt.qip.operations.rz(0.0))

    return gate

#===============================================================================#

def alter(gate):
    delta = uniform(math.pi/8, 15*math.pi/8)
    alpha = uniform(math.pi/8, 15*math.pi/8)
    theta = uniform(math.pi/8, 15* math.pi/8)
    beta = uniform(math.pi/8, 15*math.pi/8)

    rotation = random_unitary_gate(delta, alpha, theta, beta)
    alt_gate = gate*rotation

    return alt_gate

#===============================================================================#

def toffoli(c1, c2, target, circuit, indices, cats, qubits):
    """
    Generates toffoli AND gates
    """
    toff_cats = ['h', 'cx', 'tdag', 'cx', 't', 'cx', 'tdag', 'cx', 't', 't', 'cx', 'h',
                 't', 'tdag', 'cx']

    toff_inds = [target, (c2, target), target, (c1, target), target, (c2, target), target,
                 (c1, target), c2, target, (c1, c2), target, c1, c2, (c1, c2)]

    gates = [qt.qip.operations.hadamard_transform(1), qt.qip.operations.cnot(qubits, c2, target),
             tgate().dag(), qt.qip.operations.cnot(qubits, c1, target),
             tgate(), qt.qip.operations.cnot(qubits, c2, target), tgate().dag(),
             qt.qip.operations.cnot(qubits, c1, target), tgate(), tgate(),
             qt.qip.operations.cnot(qubits, c1, c2), qt.qip.operations.hadamard_transform(1),
             tgate(), tgate().dag(), qt.qip.operations.cnot(qubits, c1, c2)]

    circuit = circuit + gates
    indices = indices + toff_inds
    cats  = cats + toff_cats

    return circuit, indices, cats

#===============================================================================#

def one_qubit_adder():

    qubits = 5
    circuit, cats, indices = [], [], []

    circuit, indices, cats = toffoli(1, 2, 4, circuit, indices, cats, qubits)
    circuit, indices, cats = toffoli(0, 1, 4, circuit, indices, cats, qubits)
    circuit, indices, cats = toffoli(0, 2, 4, circuit, indices, cats, qubits)

    circuit.append(qt.qip.operations.cnot(qubits, 0, 3))
    cats.append('cx')
    indices.append((0, 3))

    circuit.append(qt.qip.operations.cnot(qubits, 1, 3))
    cats.append('cx')
    indices.append((1, 3))

    circuit.append(qt.qip.operations.cnot(qubits, 2, 3))
    cats.append('cx')
    indices.append((2, 3))

    return circuit, cats, indices, qubits

#===============================================================================#

def rot(qubits,indices_former):
    a = randint(0,qubits-2)
    b = randint(a+1,qubits-1)
    k = qt.qip.operations.cnot(qubits,a,b)

    if a == indices_former[0] and b == indices_former[1]:
        return rot(qubits, indices_former)
    else:
        return k,(a,b)

#===============================================================================#

def gen_basis_vectors(n):
    vectors = []

    for i in range(n-1):
        state = qt.basis(n, i)
        vectors.append(state)

    return vectors

#===============================================================================#

def take_places(alt_circ, cats, indices, qubits):

    dx = 0
    single_gates = ['h', 't', 'tdag']
    circuit = []

    for x in cats:
        if x in single_gates:

            temp_gate = rabbit(alt_circ[dx], qubits, indices[dx])
            circuit.append(temp_gate)
        else:
            circuit.append(alt_circ[dx])
        dx+=1

    return circuit

#===============================================================================#
def colin_mochrie(vectors, basis, circuit, cat, indices, qubits, pop, path):
    """
    Decides which indices to alter
    Finds the fidelity
    Populates test data
    """
    probabilities = []
    for i in range(pop):
        for dx in range(len(cat)):
            j = cat[dx]
            alt_circ = circuit.copy()

            if j == 'h' or j == 't' or j == 'tdag':
                alt_circ[dx] = alter(circuit[dx])

            elif j == 'cx':
                alt_circ[dx], ind_new = rot(qubits,indices[dx])

            dex, probs = 0, []
            vectors_copy = vectors.copy()

            fin_circ = take_places(alt_circ, cat, indices, qubits)

            for vector in vectors_copy:
                final = basic_b(vector, fin_circ)
                prob = dis(final, basis[dex])
                probs.append(prob)
                dex += 1

            probs.append(j + str(dx))
            probabilities.append(probs)
            print(probs)

    with open(path, 'a', newline='') as csvFile:
        for p in probabilities:
            writer = csv.writer(csvFile)
            writer.writerow(p)
        csvFile.close()

    return probabilities

#===============================================================================#

def main():

    pop = 200
    circuit, cat, indices, qubits = one_qubit_adder()
    path = 'OneQubitAdder_600.csv'
    vectors = gen_basis_vectors(2**qubits)
    basis = gen_basis_vectors(2**qubits)
    probs = colin_mochrie(vectors, basis, circuit, cat, indices, qubits, pop, path)
    KNN(probs, 0.8, 5)

    return

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))