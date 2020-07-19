import numpy as np
import qutip as qt
import math
import time
import itertools
import random
import operator
import csv

# Code with some elements taken from original simulation and some altered.
# As an attempt at simplification and consolidation of a result for REU
# Obtains two types of data which are probability vectors and fidelity data vectors
# Classifies by gate type
# B Manifold 7.14.20


# Circuit Elements needed =========================================================#


def fix_indices(indices, qubits):
    # handle problem between qutip and qiskit ordering conventions
    # which apparently don't agree

    new_indices = []
    for x in indices:
        if type(x) == tuple and len(x) == 2:
            new_indices.append((qubits - x[0] - 1, qubits - x[1] - 1))
        elif type(x) == tuple and len(x) == 3:
            new_indices.append((qubits - x[0] - 1, qubits - x[1] - 1, qubits - x[2] - 1))
        else:
            new_indices.append(qubits - x - 1)
    return new_indices


def rotation_error():
    # random 2 x 2 rotation matrices for single-qubit error

    rz1 = qt.qip.operations.rz(random.uniform(math.pi/4, -1 * math.pi / 4)).full()
    ry = qt.qip.operations.ry(random.uniform(math.pi/4, -1 * math.pi / 4)).full()
    rz2 = qt.qip.operations.rz(random.uniform(math.pi/4, -1 * math.pi / 4)).full()
    phase = qt.qip.operations.phasegate(random.uniform(math.pi/4, -1 * math.pi / 4)).full()

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

    basic_z = qt.Qobj(np.array([[1, 0],[0, -1]]))

    if alter:
        alt_z = qt.Qobj((rotation_error().dot(basic_z.full())))
        gate = qt.qip.operations.controlled_gate(alt_z, N=qubits, control=inds[0], target=inds[1], control_value=1)
        # random rotations following control gate on target
        rot1 = tensor_unitary(rotation_error(), inds[1], qubits)
        gate = rot1.dot(gate.full())
        return gate
    else:
        gate = qt.qip.operations.controlled_gate(basic_z, N=qubits, control=inds[0], target=inds[1], control_value=1)
        return gate.full()


def cnot(inds, choice, alter, qubits):
    # Alters  or builds the CNOT gates using rotations
    # Choice is a placeholder for data-gathering part

    basic_not = qt.Qobj(np.array([[0, 1],[1, 0]]))
    if alter:
        alt_not = qt.Qobj(rotation_error().dot(basic_not.full()))
        gate = qt.qip.operations.controlled_gate(alt_not, N=qubits, control=inds[0], target=inds[1], control_value=1)
        # random rotations following control gate on target
        rot1 = tensor_unitary(rotation_error(), inds[1], qubits)
        gate = rot1.dot(gate.full())
        return gate
    else:
        gate = qt.qip.operations.controlled_gate(basic_not, N=qubits, control=inds[0], target=inds[1], control_value=1)
        return gate.full()


def tensor_unitary(temp, pos, qubits):
    # Tensor up the unitaries a la qiskit

    if pos == qubits - 1:
        gate = temp
    else:
        gate = np.eye(2)
    start = gate
    for i in reversed(range(0, qubits - 1)):
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

    gate = qt.Qobj(qt.qip.operations.phasegate(-math.pi/4)*qt.qip.operations.rz(math.pi/2)\
                   *qt.qip.operations.ry(2*math.pi)*qt.qip.operations.rz(0.0))
    if alter:
        gate = rotation_error().dot(gate)
    gate = tensor_unitary(gate, pos, qubits)
    return gate


def toffoli(pos, choice, alter, qubits):
    # Alters or builds AND gates for One-Qubit adder
    # Only alters some components, so alter is False by default for some

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
    indices = [0, 1, (1, 2), (0, 1), (0), (1, 2), (0, 2)]
    inds = fix_indices(indices, qubits)
    circuit = [hadamaker(inds[0], None, False, qubits), hadamaker(inds[1], None, False, qubits), cnot(inds[2], None, False, qubits),
               cnot(inds[3], None, False, qubits), hadamaker(inds[4], None, False, qubits), cnot(inds[5], None, False, qubits),
               conv_cz(inds[6], None, False, qubits)]
    tags = ["HADAMARD", "HADAMARD", "CNOT", "CNOT", "HADAMARD", "CNOT", "CZ"]
    classes = ["HADAMARD", "CNOT", "CZ"]
    return qubits, circuit, inds, tags, classes


def ghz_circuit():

    qubits = 4
    indices = [0, (0, 1), (1, 2), (2, 3)]
    inds = fix_indices(indices, qubits)
    circuit = [hadamaker(inds[0], None, False, qubits), cnot(inds[1], None, False, qubits), cnot(inds[2], None, False, qubits),
               cnot(inds[3], None, False, qubits)]
    tags = ["HADAMARD", "CNOT", "CNOT", "CNOT"]
    classes = ["HADAMARD", "CNOT"]
    return qubits, circuit, inds, tags, classes


def w_state_circuit():

    qubits = 3
    indices = [0, 1, 2, (0, 1), 0, (1, 0), 0, 0, 1, (0, 2), (1, 2)]
    inds = fix_indices(indices, qubits)
    classes = ["RY", "X", "CNOT"]
    circuit = [ry_gate(inds[0], 0, False, qubits), paulix(inds[1], None, False, qubits), paulix(inds[2], None, False, qubits), cnot(inds[3], None, False, qubits),
               ry_gate(inds[4], 1, False, qubits), cnot(inds[5], None, False, qubits), ry_gate(inds[6], 2, False, qubits),
               paulix(inds[7], None, False, qubits), paulix(inds[8], None, False, qubits),
               cnot(inds[9], None, False, qubits), cnot(inds[10], None, False, qubits)]
    tags = ["RY", "X", "X", "CNOT", "RY", "CNOT", "RY", "X", "X", "CNOT", "CNOT"]
    return qubits, circuit, inds, tags, classes


def repeater_circuit():

    qubits = 4
    indices = [0, 2, (0, 1), (2, 3), 0, 1, 2, 3, (1, 2), 3, 1, 2, (1, 2), 1, 2, (0, 1), 0, 1, (0, 1), 0, 1, (0, 1), 1, (1, 3), 1, 3, (0, 1), 0, 1, (0, 1),
               0, 1, (0, 1)]
    inds = fix_indices(indices, qubits)
    print(inds)
    algs, alg_tags = [hadamaker, cnot], ["HADAMARD", "CNOT"]
    classes = ["HADAMARD", "CNOT"]
    tags = ["HADAMARD", "HADAMARD", "CNOT", "CNOT", "HADAMARD", "HADAMARD", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "HADAMARD",
           "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "CNOT",
            "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT", "HADAMARD", "HADAMARD", "CNOT"]
    dex, circuit = 0, []
    for x in tags:
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits))
        dex += 1
    return qubits, circuit, inds, tags, classes


def one_qubit_adder():

    qubits = 5
    indices = [(3, 2, 0), (4, 3, 0), (4, 2, 0), (4, 1), (3, 1), (2, 1)]
    inds = fix_indices(indices, qubits)
    algs, alg_tags = [toffoli, cnot], ["TOFFOLI", "CNOT"]
    classes = ["TOFFOLI", "CNOT"]
    tags = ["TOFFOLI", "TOFFOLI", "TOFFOLI", "CNOT", "CNOT", "CNOT"]
    dex, circuit = 0, []
    for x in tags:
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits))

    return qubits, circuit, inds, tags, classes


# Utility Functions ===============================================================#


def get_starting_state(qubits, choice, dex):
    # Starting state is hadamard transform on |111> for probabilities
    # And various basis states for fidelities

    if choice == 'probabilities':
        H = math.sqrt(1 / 2) * np.array([[1],[-1]])
        temp = np.copy(H)
        for x in range(qubits - 1):
            temp = np.kron(temp, H)
        return temp
    elif choice == 'fidelities':
        H_0 = math.sqrt(1 / 2) * np.array([[1], [1]])
        H_1 = math.sqrt(1 / 2) * np.array([[1], [-1]])
        choices, qs = [['000', '101', '010', '111'], ['0000', '1001', '0110', '1111'], ['00000', '10001', '01110', '11111']], [3, 4, 5]
        bin = choices[qs.index(qubits)][dex]
        hadamards = []
        for x in bin:
            if x == '0':
                hadamards.append(H_0)
            else:
                hadamards.append(H_1)
        first = hadamards[0]
        for x in hadamards[1:]:
            first = np.kron(x, first)
        return first


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
    # alter one gate

    if "Hadamard" in gate_type or "CNOT" in gate_type or "X" in gate_type or "CZ" in gate_type:
        altered_gate = alg(index, None, True, qubits)
    else:
        altered_gate = alg(index, ry_dex, True, qubits)
        ry_dex += 1
    alt[dex] = altered_gate
    state_fin = apply_circuit(psi0, alt)
    data_vec = (state_fin.T).tolist()
    return data_vec

# Data Gathering ==================================================================#

def dis(state1, state2, n):
    stateA = qt.Qobj(state1)
    stateB = qt.Qobj(state2)
    fid=qt.fidelity(stateA, stateB)
    p_0 = 1/n + (n-1)*fid/n
    # rounded to reflect available ibm accuracy
    p_0 = round(p_0, 4)
    return p_0


def check_for_duplicates(check_dups, vec):
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
    new_data_vec = list(np.around(np.real(np.conj(new_data_vec) * np.array(new_data_vec)), 4).astype(float))
    checker = check_for_duplicates(check_dups, new_data_vec)
    if checker == False and time.time() - start < 5:
        return recursive_duplicate_hunter(start, check_dups, alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)
    elif checker == False and time.time() - start >= 5:
        raise NotImplementedError
    else:
        return new_data_vec


def recursive_fidelity_hunter(start, check_dups, dex, alg, index, qubits, ry_dex, gate_type, circuit):
    # duplicate hunter for fidelities data
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


def gather_probabilities_data(pop, circuit, tags, indices, qubits, loc, classes):
    # Simulate the altered circuits
    # Round vectors so data matches available accuracy of ibm machines
    # Throw out duplicate vectors

    depth = len(circuit)
    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    alt_algs = [hadamaker, cnot, paulix, ry_gate, conv_cz, toffoli]
    check_dups, vecs = [], []

    for run in range(pop):
        ry_dex = 1
        for dex in range(depth):
            psi0 = get_starting_state(qubits, 'probabilities', None)
            gate_type = tags[dex]
            index = indices[dex]
            alt = circuit.copy()
            alg = alt_algs[gate_types.index(gate_type)]
            data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
            data_vec = list(np.around(np.real(np.conj(data_vec) * np.array(data_vec)), 4).astype(float))

            if check_for_duplicates(check_dups, data_vec):
                check_dups.append(data_vec)
                class_name = classes.index(gate_type)
                data_vec.append(class_name)
                vecs.append(data_vec)
                print(data_vec)
                #write_data(data_vec, loc)
            else:
                new_data_vec = recursive_duplicate_hunter(time.time(), check_dups, alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)
                check_dups.append(new_data_vec)
                class_name = classes.index(gate_type)
                new_data_vec.append(class_name)
                vecs.append(new_data_vec)
                print(new_data_vec)
                #write_data(new_data_vec, loc)

    last_start = get_starting_state(qubits, 'probabilities', None)
    ideal_vec = apply_circuit(last_start, circuit).T.tolist()
    ideal_vec = np.around(np.real(np.conj(ideal_vec) * np.array(ideal_vec)), 4)
    print("Ideal: ", ideal_vec.tolist()[0])
    return vecs


def diagnostic_fidelity_circuit(pop, circuit, tags, indices, qubits, loc, classes):
    # Simulate the altered circuits
    # Round vectors so data matches available accuracy of ibm machines
    # Throw out duplicate vectors

    depth = len(circuit)
    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    alt_algs = [hadamaker, cnot, paulix, ry_gate, conv_cz, toffoli]
    check_dups, vecs = [], []

    for run in range(pop):
        ry_dex = 1
        for dex in range(depth):
            gate_type = tags[dex]
            index = indices[dex]
            alg = alt_algs[gate_types.index(gate_type)]
            class_name = classes.index(gate_type)
            vector = []
            for i in range(4):
                psi0 = get_starting_state(qubits, 'fidelities', i)
                reference = psi0.copy()
                alt = circuit.copy()
                data_vec = get_altered(alt, dex, alg, index, qubits, ry_dex, psi0, gate_type)[0]
                result = dis(np.array(data_vec), reference, 2 ** qubits) # probably will have issues here.
                vector.append(result)

            if check_for_duplicates(check_dups, vector):
                check_dups.append(vector)
                vector.append(class_name)
                vecs.append(vector)
                print(vector)
                #write_data(vector, loc)
            else:
                new_data_vec = recursive_fidelity_hunter(time.time(), check_dups, dex, alg, index, qubits, ry_dex, gate_type, circuit)
                check_dups.append(new_data_vec)
                new_data_vec.append(class_name)
                vecs.append(new_data_vec)
                print(new_data_vec)
                #write_data(new_data_vec, loc)

    ideal_vector = []
    for i in range(4):
        psi0 = get_starting_state(qubits, 'fidelities', i)
        reference = psi0.copy()
        state_fin = apply_circuit(psi0, circuit)
        distance = dis(state_fin, reference, 2 ** qubits)
        ideal_vector.append(distance)
    print("Ideal: ", ideal_vector)

    return vecs


def main():

    circ_algs = [teleportation_circuit, w_state_circuit, ghz_circuit, repeater_circuit, one_qubit_adder]
    circs = ['teleport', 'wstate', 'ghz', 'repeater', 'adder']
    circ_choice = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder)'))
    metric_choice = str(input('Which metric are you using? (probabilities, fidelities)'))
    pop = int(input('How many errors per gate?'))
    loc = circ_choice + '_' + str(pop) + '.csv'
    print(loc)
    qubits, circuit, indices, tags, classes = circ_algs[circs.index(circ_choice)]()
    if metric_choice == 'probabilities':
        gather_probabilities_data(pop, circuit, tags, indices, qubits, loc, classes)
    else:
        diagnostic_fidelity_circuit(pop, circuit, tags, indices, qubits, loc, classes)


main()