import numpy as np
import qutip as qt
import math
import time
import itertools
import random
import operator
import csv

# Does the same thing as ClassicalSim, but with the original plan to identify regions of error with all the gates indexed
# 7.25.20


# 7.24 : qutips ordering convention is backwards from qiskits, but this got
# confounded with the tensoring here, so it had to be modified

# Circuit Elements needed =========================================================#


def rotation_error():
    # random 2 x 2 rotation matrices for single-qubit error
    # altered to smaller angles since ibm data failed to train well on larger angles

    rz1 = qt.qip.operations.rz(random.uniform(0, math.pi/4)).full()
    ry = qt.qip.operations.ry(random.uniform(0, math.pi/4)).full()
    rz2 = qt.qip.operations.rz(random.uniform(0, math.pi/4)).full()
    phase = qt.qip.operations.phasegate(random.uniform(0, math.pi/4)).full()

    gate = phase.dot(rz1.dot(ry.dot(rz2)))
    return gate


def pick_alter(original, altered, alter):
    if alter:
        return altered
    else:
        return original


def conv_cz(inds, choice, alter, qubits):
    # Alters or builds the Control - Z gate using rotations
    # Choice is a placeholder for data-gathering part
    # Changed rotations to help machine learning fit

    basic_z = qt.Qobj(np.array([[1, 0],[0, -1]]))
    # qiskit-qutip indexing issue
    ind0 = qubits - inds[0] - 1
    ind1 = qubits - inds[1] - 1

    if alter:
        # random rotations applied to target qubit
        alt_z = qt.Qobj((rotation_error().dot(basic_z.full())))
        gate = qt.qip.operations.controlled_gate(alt_z, N=qubits, control=ind0, target=ind1, control_value=1)
        # random rotations applied to control qubit
        rot2 = tensor_unitary(rotation_error(), ind0, qubits)
        gate = rot2.dot(gate.full())
        return gate
    else:
        gate = qt.qip.operations.controlled_gate(basic_z, N=qubits, control=ind0, target=ind1, control_value=1)
        return gate.full()


def cnot(inds, choice, alter, qubits):
    # Alters  or builds the CNOT gates using rotations
    # Choice is a placeholder for data-gathering part
    # Changed rotations to help machine learning fit

    basic_not = qt.Qobj(np.array([[0, 1],[1, 0]]))
    # qiskit-qutip indexing issue
    ind0 = qubits - inds[0] - 1
    ind1 = qubits - inds[1] - 1

    if alter:
        # random rotations on target qubit
        alt_not = qt.Qobj(rotation_error().dot(basic_not.full()))
        gate = qt.qip.operations.controlled_gate(alt_not, N=qubits, control=ind0, target=ind1, control_value=1)
        # random rotations on control qubit
        rot2 = tensor_unitary(rotation_error(), ind0, qubits)
        gate = rot2.dot(gate.full())
        return gate
    else:
        gate = qt.qip.operations.controlled_gate(basic_not, N=qubits, control=ind0, target=ind1, control_value=1)
        return gate.full()


def tensor_unitary(temp, pos, qubits):
    # Tensor up the unitaries
    # This is the main part that got bungled

    if pos == 0:
        gate = temp
    else:
        gate = np.eye(2)
    start = gate
    for i in range(1, qubits):
        if i == pos:
            start = np.kron(temp, start)
        else:
            start = np.kron(np.eye(2), start)
    return start


def hadamaker(pos, choice, alter, qubits):
    # Alters or builds Hadamards
    # Choice is a placeholder for data-gathering part

    had_basic = np.array([[1, 1], [1, -1]]) * math.sqrt(1 / 2)
    temp = pick_alter(had_basic, rotation_error().dot(had_basic), alter)
    gate = tensor_unitary(temp, pos, qubits)
    return gate


def paulix(pos, choice, alter, qubits):
    # Alters or builds NOT gates
    # Choice is a placeholder for data-gathering part

    not_basic = np.array([[0, 1], [1, 0]])
    temp = pick_alter(not_basic, rotation_error().dot(not_basic), alter)
    gate = tensor_unitary(temp, pos, qubits)
    return gate


def ry_gate(pos, choice, alter, qubits):
    # Alters or builds ry unitaries for W-state

    choices = [qt.qip.operations.ry(-1.23096).full(),
               qt.qip.operations.ry(np.pi / 4).full(),
               qt.qip.operations.ry(-1 * np.pi / 4).full()]
    temp = pick_alter(choices[choice], rotation_error().dot(choices[choice]), alter)
    gate = tensor_unitary(temp, pos, qubits)
    return gate


def tgate(pos, choice, alter, qubits):
    # Alters or builds T gates for Adder

    gate = qt.Qobj(qt.qip.operations.phasegate(-math.pi/4)*qt.qip.operations.rz(math.pi/2)\
                   *qt.qip.operations.ry(2*math.pi)*qt.qip.operations.rz(0.0))
    if alter:
        gate = rotation_error().dot(gate)
    gate = tensor_unitary(gate, pos, qubits)
    return gate


def toffoli(pos, choice, alter, qubits):
    # Alters or builds AND gates for One-Qubit adder
    # Only alters some components, so alter is False by default for some
    # Still might split up

    c1, c2, target = pos[0], pos[1], pos[2]
    gates = [hadamaker(target, choice, alter, qubits), cnot((c2, target), choice, alter, qubits),
             np.conj(tgate(target, choice, alter, qubits).T), cnot((c1, target), choice, False, qubits),
             tgate(target, choice, alter, qubits), cnot((c2, target), choice, False, qubits), np.conj(tgate(target, choice, False, qubits)).T,
             cnot((c1, target), choice, alter, qubits), tgate(c2, choice, False, qubits), tgate(target, choice, False, qubits),
             cnot((c1, c2), choice, False, qubits), hadamaker(target, choice, alter, qubits), tgate(c1, choice, alter, qubits),
             np.conj(tgate(c2, choice, alter, qubits)).T, cnot((c1, c2), choice, False, qubits)]
    begin = gates[-1]
    for x in reversed(gates[:-1]):
        begin = np.dot(x, begin)

    return begin


# Relevant Circuits ===============================================================#

def teleportation_circuit():
    # Takes the hadamard off qubit 0, to teleport state |0> to qubit 2
    # Therefore has extra gate in the beginning

    qubits = 3
    inds = [0, 1, (1, 2), (0, 1), 0, (1, 2), (0, 2)]
    circuit = [hadamaker(inds[0], None, False, qubits), hadamaker(inds[1], None, False, qubits), cnot(inds[2], None, False, qubits),
               cnot(inds[3], None, False, qubits), hadamaker(inds[4], None, False, qubits), cnot(inds[5], None, False, qubits),
               conv_cz(inds[6], None, False, qubits)]
    tags = ["HADAMARD1", "HADAMARD2", "CNOT1", "CNOT2", "HADAMARD3", "CNOT3", "CZ1"]
    classes = ["HADAMARD", "CNOT", "CZ"]
    return qubits, circuit, inds, tags, classes


def ghz_circuit():

    qubits = 4
    inds = [0, (0, 1), (1, 2), (2, 3)]
    circuit = [hadamaker(inds[0], None, False, qubits), cnot(inds[1], None, False, qubits), cnot(inds[2], None, False, qubits),
               cnot(inds[3], None, False, qubits)]
    tags = ["HADAMARD1", "CNOT1", "CNOT2", "CNOT3"]
    classes = ["HADAMARD", "CNOT"]
    return qubits, circuit, inds, tags, classes


def w_state_circuit():

    qubits = 3
    inds = [0, 1, 2, (0, 1), 0, (1, 0), 0, 0, 1, (0, 2), (1, 2)]
    classes = ["RY", "X", "CNOT"]
    circuit = [ry_gate(inds[0], 0, False, qubits), paulix(inds[1], None, False, qubits), paulix(inds[2], None, False, qubits), cnot(inds[3], None, False, qubits),
               ry_gate(inds[4], 1, False, qubits), cnot(inds[5], None, False, qubits), ry_gate(inds[6], 2, False, qubits),
               paulix(inds[7], None, False, qubits), paulix(inds[8], None, False, qubits),
               cnot(inds[9], None, False, qubits), cnot(inds[10], None, False, qubits)]
    tags = ["RY1", "X1", "X2", "CNOT1", "RY2", "CNOT2", "RY3", "X3", "X4", "CNOT3", "CNOT4"]
    return qubits, circuit, inds, tags, classes


def repeater_circuit():

    qubits = 4
    inds = [0, 2, (0, 1), (2, 3), 0, 1, 2, 3, (1, 2), 3, 1, 2, (1, 2), 1, 2, (0, 1), 0, 1, (0, 1), 0, 1, (0, 1), 1, (1, 3), 1, 3, (0, 1), 0, 1, (0, 1),
               0, 1, (0, 1)]
    algs, alg_tags = [hadamaker, cnot], ["HADAMARD", "CNOT"]
    classes = ["HADAMARD", "CNOT"]
    tags = ["HADAMARD1", "HADAMARD2", "CNOT1", "CNOT2", "HADAMARD3", "HADAMARD4", "HADAMARD5", "HADAMARD6", "CNOT3", "HADAMARD7", "HADAMARD8", "HADAMARD9",
           "CNOT4", "HADAMARD10", "HADAMARD11", "CNOT5", "HADAMARD12", "HADAMARD13", "CNOT6", "HADAMARD14", "HADAMARD15", "CNOT7", "HADAMARD16", "CNOT8",
            "HADAMARD17", "HADAMARD18", "CNOT9", "HADAMARD19", "HADAMARD20", "CNOT10", "HADAMARD21", "HADAMARD22", "CNOT11"]
    type_tags = ["HADAMARD", "HADAMARD", "CNOT", "CNOT", "HADAMARD", "HADAMARD", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "HADAMARD",
            "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "CNOT",
            "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT"]
    dex, circuit = 0, []
    for x in type_tags:
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits))
        dex += 1
    return qubits, circuit, inds, tags, classes


def one_qubit_adder():

    qubits = 5
    inds = [(3, 2, 0), (4, 3, 0), (4, 2, 0), (4, 1), (3, 1), (2, 1)]
    algs, alg_tags = [toffoli, cnot], ["TOFFOLI", "CNOT"]
    classes = ["TOFFOLI", "CNOT"]
    tags = ["TOFFOLI1", "TOFFOLI2", "TOFFOLI3", "CNOT1", "CNOT2", "CNOT3"]
    type_tags = ["TOFFOLI", "TOFFOLI", "TOFFOLI", "CNOT", "CNOT", "CNOT"]
    dex, circuit = 0, []
    for x in type_tags:
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits))

    return qubits, circuit, inds, tags, classes


# Utility Functions ===============================================================#


def get_starting_state(qubits, choice, dex):
    # Starting state is hadamard transform on |111> for probabilities
    # And various basis states for fidelities

    choices, qs = [['000', '101', '010', '111'], ['0000', '1001', '0110', '1111'], ['00000', '10001', '01110', '11111']], [3, 4, 5]

    if choice == 'probabilities':
        H = math.sqrt(1 / 2) * np.array([[1],[-1]])
        temp = np.copy(H)
        for x in range(qubits - 1):
            temp = np.kron(temp, H)
        return temp

    elif choice == 'fidelities':
        bin = choices[qs.index(qubits)][dex]
        h = np.array([[1, 1],[1, -1]]) * math.sqrt(1 / 2)
        hadamard = h.copy()
        for x in range(qubits - 1):
            hadamard = np.kron(hadamard, h)
        index = int(bin, 2)
        starting_vector = np.zeros((2 ** qubits, 1))
        starting_vector[index, 0] += 1
        starting_vector = hadamard.dot(starting_vector)
        return starting_vector


def apply_circuit(state, circuit):

    state_fin = circuit[0].dot(state)
    for i in range(1, len(circuit)):
        state_fin = circuit[i].dot(state_fin)
    return state_fin


def write_data(vec, loc):
    with open(loc,'a',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(vec)
    csvFile.close()


def get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type):
    # Alter one gate of chosen type, find ending state
    if "Hadamard" in gate_type or "CNOT" in gate_type or "X" in gate_type or "CZ" in gate_type or "TOFFOLI" in gate_type:
        altered_gate = alg(index, None, True, qubits)
    else:
        altered_gate = alg(index, ry_dex, True, qubits)
        ry_dex += 1
    alt[dex] = altered_gate
    state_fin = apply_circuit(psi0, alt)
    data_vec = (state_fin.T).tolist()
    return data_vec

# Data Gathering ==================================================================#

def dis(state1,state2, n):
    # had to verify what the original formula was doing
    # find probability of |0> for ancilla
    overlap = list(np.real(np.dot(np.conj(np.array(state1).T), np.array(state2)) * np.dot(np.conj(np.array(state2).T), state1)))[0]
    p_0 = 0.5 + 0.5 * overlap
    return p_0


def check_for_duplicates(check_dups, vec):
    # check a current vector against other found vectors to prevent duplicates
    for x in check_dups:
        eqs = 0
        for y in range(len(vec)):
            if vec[y] == x[y]:
                eqs += 1
        if eqs == len(vec):
            return False
    return True


def recursive_duplicate_hunter(start, check_dups, alt, dex, alg, index, qubits, ry_dex, psi0, gate_type):
    # duplicate hunter for probabilities data
    new_data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
    new_data_vec = list(np.real(np.conj(new_data_vec) * np.array(new_data_vec)).astype(float))
    checker = check_for_duplicates(check_dups, new_data_vec)
    if checker == False and time.time() - start < 5:
        return recursive_duplicate_hunter(start, check_dups, alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)
    elif checker == False and time.time() - start >= 5:
        raise NotImplementedError
    else:
        return new_data_vec


def recursive_fidelity_hunter(start, check_dups, dex, alg, index, qubits, ry_dex, gate_type, circuit):
    # duplicate hunter for fidelities data
    # repeatedly looks for new vectors if the options get sparse
    vector = []
    alt = circuit.copy()
    for i in range(4):
        psi0 = get_starting_state(qubits, 'fidelities', i)
        reference = psi0.copy()
        new_data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
        result = dis(np.array(new_data_vec), reference, 2 ** qubits)
        vector.append(result)
    checker = check_for_duplicates(check_dups, vector)
    if checker == False and time.time() - start < 5:
        return recursive_fidelity_hunter(start, check_dups, dex, alg, index, qubits, ry_dex, gate_type, circuit)
    elif checker == False and time.time() - start >= 5:
        raise NotImplementedError
    else:
        return vector


def get_global_gate_type(gate_type):
    # Take out the index to slice out alt_algs

    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    for x in gate_types:
        if x in gate_type:
            return x


def gather_probabilities_data(pop, circuit, tags, indices, qubits, loc, classes):
    # Simulate the altered circuits
    # Throw out duplicate vectors

    depth = len(circuit)
    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    alt_algs = [hadamaker, cnot, paulix, ry_gate, conv_cz, toffoli]
    check_dups, vecs = [], []

    for run in range(pop):
        # run through an error on each individual gate
        ry_dex = 1
        for dex in range(depth):
            # find final probabilities vector with altered gate
            psi0 = get_starting_state(qubits, 'probabilities', None)
            gate_type = tags[dex]
            index = indices[dex]
            alt = circuit.copy()
            global_gate_type = get_global_gate_type(gate_type)
            alg = alt_algs[gate_types.index(global_gate_type)]
            data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
            data_vec = list(np.real(np.conj(data_vec) * np.array(data_vec)).astype(float))

            if check_for_duplicates(check_dups, data_vec):
                check_dups.append(data_vec)
                data_vec.append(dex)
                vecs.append(data_vec)
                print(data_vec)
                write_data(data_vec, loc)
            else:
                new_data_vec = recursive_duplicate_hunter(time.time(), check_dups, alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)
                check_dups.append(new_data_vec)
                new_data_vec.append(dex)
                vecs.append(new_data_vec)
                print(new_data_vec)
                write_data(new_data_vec, loc)

    # find the ideal vector for comparison
    last_start = get_starting_state(qubits, 'probabilities', None)
    ideal_vec = apply_circuit(last_start, circuit).T.tolist()
    ideal_vec = np.real(np.conj(ideal_vec) * np.array(ideal_vec))
    print("Ideal: ", ideal_vec.tolist()[0])
    return vecs


def diagnostic_fidelity_circuit(pop, circuit, tags, indices, qubits, loc, classes):
    # Simulate the altered circuits
    # Throw out duplicate vectors

    depth = len(circuit)
    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    alt_algs = [hadamaker, cnot, paulix, ry_gate, conv_cz, toffoli]
    check_dups, vecs = [], []

    for run in range(pop):
        # run through an error on each individual gate
        ry_dex = 1
        for dex in range(depth):
            # find vector of p(|0>) from the four chosen basis states with altered gate at index dex
            gate_type = tags[dex]
            index = indices[dex]
            global_gate_type = get_global_gate_type(gate_type)
            alg = alt_algs[gate_types.index(global_gate_type)]
            vector = []
            for i in range(4):
                psi0 = get_starting_state(qubits, 'fidelities', i)
                reference = psi0.copy()
                alt = circuit.copy()
                data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
                result = dis(np.array(data_vec), reference, 2 ** qubits)
                vector.append(result)

            if check_for_duplicates(check_dups, vector):
                check_dups.append(vector)
                vector.append(dex)
                vecs.append(vector)
                print(vector)
                write_data(vector, loc)
            else:
                new_data_vec = recursive_fidelity_hunter(time.time(), check_dups, dex, alg, index, qubits, ry_dex, gate_type, circuit)
                check_dups.append(new_data_vec)
                new_data_vec.append(dex)
                vecs.append(new_data_vec)
                print(new_data_vec)
                write_data(new_data_vec, loc)

    # find ideal vector for comparison
    ideal_vector = []
    for i in range(4):
        psi0 = get_starting_state(qubits, 'fidelities', i)
        reference = psi0.copy()
        state_fin = apply_circuit(psi0, circuit)
        distance = dis(state_fin, reference, 2 ** qubits)
        ideal_vector.append(list(distance)[0])
    print("Ideal: ", ideal_vector)

    return vecs


def main():

    circ_algs = [teleportation_circuit, w_state_circuit, ghz_circuit, repeater_circuit, one_qubit_adder]
    circs = ['teleport', 'wstate', 'ghz', 'repeater', 'adder']
    # Ask many questions
    circ_choice = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder)'))
    metric_choice = str(input('Which metric are you using? (probabilities, fidelities)'))
    pop = int(input('How many errors per gate?'))
    loc = circ_choice + '_' + str(pop) + metric_choice + '_allgates' + '_.csv'
    # print stored location
    print(loc)

    # fetch the circuit stuff
    qubits, circuit, indices, tags, classes = circ_algs[circs.index(circ_choice)]()

    # gather data
    if metric_choice == 'probabilities':
        gather_probabilities_data(pop, circuit, tags, indices, qubits, loc, classes)
    else:
        diagnostic_fidelity_circuit(pop, circuit, tags, indices, qubits, loc, classes)


main()