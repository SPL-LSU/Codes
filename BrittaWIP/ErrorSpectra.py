# Generate an image of fidelity variations with broken gates
# Black will be regions of high discrepancy, white is ideal
# For science
# B Manifold 7. 19. 20

import numpy as np
import qutip as qt
import math
import time
import itertools
import random
import operator
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy


# Circuit Elements needed =========================================================#
severity_range = 200
severity_increment = math.pi / severity_range

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


def rotation_error(severity):
    # 2 x 2 rotation matrices for single-qubit error

    severity = severity * severity_increment

    rz1 = qt.qip.operations.rz(severity).full()
    ry = qt.qip.operations.ry(severity).full()
    rz2 = qt.qip.operations.rz(severity).full()
    phase = qt.qip.operations.phasegate(severity).full()

    gate = phase.dot(rz1.dot(ry.dot(rz2)))
    return gate


def pick_alter(original, altered, alter):
    if alter:
        return altered
    else:
        return original


def conv_cz(inds, choice, alter, qubits, severity):
    # Alters or builds the Control - Z gate using rotations
    # Choice is a placeholder for data-gathering part

    basic_z = qt.Qobj(np.array([[1, 0],[0, -1]]))

    if alter:
        alt_z = qt.Qobj((rotation_error(severity).dot(basic_z.full())))
        gate = qt.qip.operations.controlled_gate(alt_z, N=qubits, control=inds[0], target=inds[1], control_value=1)
        # rotations following control gate on target
        rot1 = tensor_unitary(rotation_error(severity), inds[1], qubits)
        gate = rot1.dot(gate.full())
        rot2 = tensor_unitary(rotation_error(severity), inds[0], qubits)
        gate = rot2.dot(gate)
        return gate
    else:
        gate = qt.qip.operations.controlled_gate(basic_z, N=qubits, control=inds[0], target=inds[1], control_value=1)
        return gate.full()


def cnot(inds, choice, alter, qubits, severity):
    # Alters  or builds the CNOT gates using rotations
    # Choice is a placeholder for data-gathering part

    basic_not = qt.Qobj(np.array([[0, 1],[1, 0]]))
    if alter:
        alt_not = qt.Qobj(rotation_error(severity).dot(basic_not.full()))
        gate = qt.qip.operations.controlled_gate(alt_not, N=qubits, control=inds[0], target=inds[1], control_value=1)
        # rotations following control gate on target
        rot1 = tensor_unitary(rotation_error(severity), inds[1], qubits)
        gate = rot1.dot(gate.full())
        rot2 = tensor_unitary(rotation_error(severity), inds[0], qubits)
        gate = rot2.dot(gate)
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


def hadamaker(pos, choice, alter, qubits, severity):
    # Alters or builds Hadamards
    # Choice is a placeholder for data-gathering part

    had_basic = np.array([[1, 1], [1, -1]]) * math.sqrt(1 / 2)
    temp = pick_alter(had_basic, rotation_error(severity).dot(had_basic), alter)
    gate = tensor_unitary(temp, pos, qubits)
    return gate


def paulix(pos, choice, alter, qubits, severity):
    # Alters or builds NOT gates
    # Choice is a placeholder for data-gathering part

    not_basic = np.array([[0, 1], [1, 0]])
    temp = pick_alter(not_basic, rotation_error(severity).dot(not_basic), alter)

    gate = tensor_unitary(temp, pos, qubits)
    return gate


def ry_gate(pos, choice, alter, qubits, severity):
    # Alters or builds ry unitaries for W-state

    choices = [qt.qip.operations.ry(-1.23096).full(),
               qt.qip.operations.ry(np.pi / 4).full(),
               qt.qip.operations.ry(-1 * np.pi / 4).full()]
    temp = pick_alter(choices[choice], rotation_error(severity).dot(choices[choice]), alter)

    gate = tensor_unitary(temp, pos, qubits)
    return gate


def tgate(pos, choice, alter, qubits, severity):

    gate = qt.Qobj(qt.qip.operations.phasegate(-math.pi/4)*qt.qip.operations.rz(math.pi/2)\
                   *qt.qip.operations.ry(2*math.pi)*qt.qip.operations.rz(0.0))
    if alter:
        gate = rotation_error(severity).dot(gate)
    gate = tensor_unitary(gate, pos, qubits)
    return gate


def toffoli(pos, choice, alter, qubits, severity):
    # Alters or builds AND gates for One-Qubit adder
    # Only alters some components, so alter is False by default for some

    s = severity
    c1, c2, target = pos[0], pos[1], pos[2]
    gates = [hadamaker(target, choice, alter, qubits, s), cnot((c2, target), choice, alter, qubits, s),
             np.conj(tgate(target, choice, alter, qubits, s).T), cnot((c1, target), choice, False, qubits, s),
             tgate(target, choice, alter, qubits, s), cnot((c2, target), choice, False, qubits, s), np.conj(tgate(target, choice, False, qubits, s)).T,
             cnot((c1, target), choice, alter, qubits, s), tgate(c2, choice, False, qubits, s), tgate(target, choice, False, qubits, s),
             cnot((c1, c2), choice, False, qubits, s), hadamaker(target, choice, alter, qubits, s), tgate(c1, choice, alter, qubits, s),
             np.conj(tgate(c2, choice, alter, qubits, s)).T, cnot((c1, c2), choice, False, qubits, s)]
    begin = gates[-1]
    for x in reversed(gates[:-1]):
        begin = np.dot(x, begin)

    return begin


# Relevant Circuits ===============================================================#
# Severity for the base circuits is 0 since it won't be used

def teleportation_circuit():
    # Takes the hadamard off qubit 0, to teleport state |0> to qubit 2
    # Therefore has extra gate in the beginning
    qubits = 3
    indices = [0, 1, (1, 2), (0, 1), (0), (1, 2), (0, 2)]
    inds = fix_indices(indices, qubits)
    circuit = [hadamaker(inds[0], None, False, qubits, 0), hadamaker(inds[1], None, False, qubits, 0), cnot(inds[2], None, False, qubits, 0),
               cnot(inds[3], None, False, qubits, 0), hadamaker(inds[4], None, False, qubits, 0), cnot(inds[5], None, False, qubits, 0),
               conv_cz(inds[6], None, False, qubits, 0)]
    tags = ["HADAMARD", "HADAMARD", "CNOT", "CNOT", "HADAMARD", "CNOT", "CZ"]
    classes = ["HADAMARD", "CNOT", "CZ"]
    return qubits, circuit, inds, tags, classes


def ghz_circuit():

    qubits = 4
    indices = [0, (0, 1), (1, 2), (2, 3)]
    inds = fix_indices(indices, qubits)
    circuit = [hadamaker(inds[0], None, False, qubits, 0), cnot(inds[1], None, False, qubits, 0), cnot(inds[2], None, False, qubits, 0),
               cnot(inds[3], None, False, qubits, 0)]
    tags = ["HADAMARD", "CNOT", "CNOT", "CNOT"]
    classes = ["HADAMARD", "CNOT"]
    return qubits, circuit, inds, tags, classes


def w_state_circuit():

    qubits = 3
    indices = [0, 1, 2, (0, 1), 0, (1, 0), 0, 0, 1, (0, 2), (1, 2)]
    inds = fix_indices(indices, qubits)
    classes = ["RY", "X", "CNOT"]
    circuit = [ry_gate(inds[0], 0, False, qubits, 0), paulix(inds[1], None, False, qubits, 0), paulix(inds[2], None, False, qubits, 0),
               cnot(inds[3], None, False, qubits, 0), ry_gate(inds[4], 1, False, qubits, 0), cnot(inds[5], None, False, qubits, 0),
               ry_gate(inds[6], 2, False, qubits, 0), paulix(inds[7], None, False, qubits, 0), paulix(inds[8], None, False, qubits, 0),
               cnot(inds[9], None, False, qubits, 0), cnot(inds[10], None, False, qubits, 0)]
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
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits, 0))
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
        circuit.append(algs[alg_tags.index(x)](inds[dex], None, False, qubits, 0))

    return qubits, circuit, inds, tags, classes


# Utility Functions ===============================================================#


def get_starting_state(bin):
    # Starting state is hadamard transform

    H_0 = math.sqrt(1 / 2) * np.array([[1], [1]])
    H_1 = math.sqrt(1 / 2) * np.array([[1], [-1]])
    hadamards = []
    for x in bin:
        if x == '0':
            hadamards.append(H_0)
        else:
            hadamards.append(H_1)
    last = hadamards[-1]
    for x in reversed(hadamards[:-1]):
        last = np.kron(x, last)
    return last


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


def get_altered(alg, psi0, qubits, circuit, indices, gate_type, severity, tags):
    # alter a class of gates with a particular severity

    alt = []
    ry_dex = 0
    for x in range(len(circuit)):
        index = indices[x]
        if tags[x] == gate_type:
            if gate_type != "RY":
                altered_gate = alg(index, None, True, qubits, severity)
                alt.append(altered_gate)
            else:
                altered_gate = alg(index, ry_dex, True, qubits, severity)
                alt.append(altered_gate)
                ry_dex += 1
        else:
            alt.append(circuit[x])
    state_fin = apply_circuit(psi0, alt)
    data_vec = (state_fin.T).tolist()

    return data_vec

# Data Gathering ==================================================================#

def dis(state1, state2, n):
    stateA = qt.Qobj(state1)
    stateB = qt.Qobj(state2)
    fid=qt.fidelity(stateA, stateB)
    p_0 = 1/n + (n-1)*fid/n
    return p_0


def to_image(arrays, ideal_array, classes, circ_choice, metric_choice):

    difference_arrays = []
    for x in range(len(arrays)):
        difference_arrays.append(np.ones_like(arrays[x]) - np.abs(arrays[x] - ideal_array))

    if len(classes) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(str(metric_choice) + ' data variation for ' + str(circ_choice) + ' circuit', fontsize=15)
        axes = [ax1, ax2]
        for i in range(len(axes)):
            axes[i].set_title(classes[i])
            axes[i].set_title(classes[i])
            axes[i].imshow(difference_arrays[i], cmap='gray', aspect='auto')
            axes[i].set_ylabel(r'$\longleftarrow\theta$')
            axes[i].set_xlabel(r'$|\psi_{i}\rangle$')
            axes[i].set_yticks([])
            axes[i].set_xticks([])
    elif len(classes) == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(str(metric_choice) + ' data variation for ' + str(circ_choice) + ' circuit', fontsize=15)
        axes = [ax1, ax2, ax3]
        for i in range(len(axes)):
            axes[i].set_title(classes[i])
            axes[i].imshow(difference_arrays[i], cmap='gray', aspect='auto')
            axes[i].set_ylabel(r'$\longleftarrow\theta$')
            axes[i].set_xlabel(r'$|\psi_{i}\rangle$')
            axes[i].set_yticks([])
            axes[i].set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def to_projection(arrays, ideal_array, circ_choice, metric_choice, classes):
    # subtract this image from ideal to plot the variation

    difference_arrays = []
    for x in range(len(arrays)):
        difference_arrays.append(np.ones_like(arrays[x]) - np.abs(arrays[x] - ideal_array))

    for i in range(len(difference_arrays)):
        image = difference_arrays[i]
        xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle(str(metric_choice) + ' data variation for ' + str(circ_choice) + ' circuit (' + classes[i] + ')', fontsize=15)
        ax.plot_surface(xx, yy, image, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
        ax.set_ylabel(r'$|\psi_{i}\rangle \longrightarrow $')
        ax.set_xlabel(r'$\theta\longrightarrow$')
        ax.set_yticks([])
        ax.set_xticks([])
        plt.show()

    return


def diagnostic_fidelity_image():
    # Plot images of changing error (from 0 to pi rotations)for different classes in a circuit
    circ_algs = [teleportation_circuit, w_state_circuit, ghz_circuit, repeater_circuit, one_qubit_adder]
    circs = ['teleport', 'wstate', 'ghz', 'repeater', 'adder']
    circ_choice = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder)'))
    metric_choice = str(input('Which metric?(fidelities, probabilities)'))
    loc = circ_choice + 'image' +  '_.csv' # filetype
    print(loc)
    qubits, circuit, indices, tags, classes = circ_algs[circs.index(circ_choice)]()
    images = []
    gate_types = ["HADAMARD", "CNOT", "X", "RY", "CZ", "TOFFOLI"]
    alt_algs = [hadamaker, cnot, paulix, ry_gate, conv_cz, toffoli]
    for gate_type in classes:
        image = []
        for severity in range(severity_range):
            alg = alt_algs[gate_types.index(gate_type)]
            if metric_choice == 'fidelities':
                vector = []
                for i in range(2 ** qubits):
                    bin = np.binary_repr(i).zfill(qubits)
                    psi0 = get_starting_state(bin)
                    reference = psi0.copy()
                    data_vec = get_altered(alg, psi0, qubits, circuit, indices, gate_type, severity, tags)
                    result = dis(np.array(data_vec), reference, 2 ** qubits)
                    vector.append(result)
                print(vector)
                image.append(vector)
            elif metric_choice == 'probabilities':
                psi0 = get_starting_state('0' * qubits)
                data_vec = get_altered(alg, psi0, qubits, circuit, indices, gate_type, severity, tags)[0]
                result = list(np.real(np.conj(data_vec) * np.array(data_vec)).astype(float))
                image.append(result)
        images.append(np.array(image))


    # generate the ideal image to take the difference
    ideal_vector = []
    if metric_choice == 'fidelities':
        for i in range(2 ** qubits):
            bin = np.binary_repr(i).zfill(qubits)
            psi0 = get_starting_state(bin)
            reference = psi0.copy()
            state_fin = apply_circuit(psi0, circuit)
            distance = dis(state_fin, reference, 2 ** qubits)
            ideal_vector.append(distance)
    elif metric_choice == 'probabilities':
        psi0 =  get_starting_state('0' * qubits)
        state_fin = apply_circuit(psi0, circuit)[0]
        result = list(np.real(np.conj(state_fin) * np.array(state_fin)).astype(float))
        ideal_vector += result

    # ideal array for one image
    ideal_array = []
    for x in range(severity_range):
        ideal_array.append(ideal_vector)

    ideal_array = np.array(ideal_array)

    to_image(images, ideal_array, classes, circ_choice, metric_choice)
    #to_projection(images, ideal_array, circ_choice, metric_choice, classes)

diagnostic_fidelity_image()