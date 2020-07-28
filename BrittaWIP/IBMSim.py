# code to gather noisy simulator test data for circuits
# choice of probabilities data or fidelities data
# B Manifold updated 7.24.20

import numpy as np
import qiskit as q
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ
import csv
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
import math
from qiskit.providers.aer.noise.device import models
from qiskit.extensions import UnitaryGate


IBMQ.enable_account("84fd70bd5cf0a01b623843a1e228a627e78c0c1c6d0f3a0527b1240c702630ee36bb2b71c85c2c8148011dfbc3a4400520836eaf69d5c91d87e5a9ccb3f85f75")
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
simulator = Aer.get_backend('qasm_simulator')

def initialize_circuit(qubits):
    # Initial state is Hadamard transform on |111..>
    # Initial circuit for probabilities data

    circuit = QuantumCircuit(qubits, qubits)
    for x in range(qubits):
        circuit.x(x)
        circuit.h(x)
    return circuit


def initialize_diagnostic_circuit(qubits, dex):
    # Initial state is a Hadamard transform
    # Initial circuit for fidelities data

    choices, qs = [['000', '101', '010', '111'], ['0000', '1001', '0110', '1111'], ['00000', '10001', '01110', '11111']], [3, 4, 5]
    bin = choices[qs.index(qubits)][dex]

    qc = QuantumCircuit(2 * qubits + 1, 1)
    pos = 0
    for x in reversed(bin):
        if x == '1':
            qc.x(pos + 1)
            qc.x(pos + qubits + 1)
        pos += 1
    for x in range(1, 2 * qubits + 1):
        # hadamards on all logical qubits
        qc.h(x)
    return qc


def fredkin3(qc, control, t1, t2):
    # this had an issue, don't know what it was yet.
    # but ideal vectors weren't equal for classical/ibm data
    return qc.cswap(control, t1, t2)


def GHZ_circuit(qc, choice):

    if choice == 'probabilities':
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        for x in range(4):
            qc.measure(x, x)
    elif choice == 'fidelities':
        qubits = 4
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)

        # swap test
        qc.h(0)
        for x in range(1, qubits + 1):
            qc.barrier()
            fredkin3(qc, 0, x, x + qubits)
            qc.barrier()
        qc.h(0)

        qc.measure(0, 0)

def deutsch_jozsa_circuit(qc, choice):

    if choice == 'probabilities':

        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.x(4)
        qc.h(4)
        qc.x(0)
        qc.x(2)
        qc.cx(0, 4)
        qc.x(0)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.x(2)
        qc.cx(3, 4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        for x in range(5):
            qc.measure(x, x)

    elif choice == 'fidelities':
        qubits = 5
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.h(4)
        qc.x(5)
        qc.h(5)
        qc.x(1)
        qc.x(3)
        qc.cx(1, 5)
        qc.x(1)
        qc.cx(2, 5)
        qc.cx(3, 5)
        qc.x(3)
        qc.cx(4, 5)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.h(4)

        # swap test
        qc.h(0)
        for x in range(1, qubits + 1):
            qc.barrier()
            fredkin3(qc, 0, x, x + qubits)
            qc.barrier()
        qc.h(0)

        qc.measure(0, 0)

def teleportation_circuit(qc, choice):

    if choice == 'probabilities':
        qc.h(0)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(1, 2)
        qc.cz(0, 2)
        for x in range(3):
            qc.measure(x, x)
    elif choice == 'fidelities':
        qubits = 3
        qc.h(1)
        qc.h(2)
        qc.cx(2, 3)
        qc.cx(1, 2)
        qc.h(1)
        qc.cx(2, 3)
        qc.cz(1, 3)

        # swap test
        qc.h(0)
        for x in range(1, qubits + 1):
            qc.barrier()
            fredkin3(qc, 0, x, x + qubits)
            qc.barrier()
        qc.h(0)

        qc.measure(0, 0)


def wstate_circuit(w3, choice):

    if choice == 'probabilities':
        w3.ry(-1.23096, 0)
        w3.x(1)
        w3.x(2)
        w3.cx(0, 1)
        w3.ry(np.pi / 4, 0)
        w3.cx(1, 0)
        w3.ry(-1 * np.pi / 4, 0)
        w3.x(0)
        w3.x(1)
        w3.cx(0, 2)
        w3.cx(1, 2)
        for x in range(3):
            w3.measure(x, x)
    elif choice == 'fidelities':
        qubits = 3
        w3.ry(-1.23096, 1)
        w3.x(2)
        w3.x(3)
        w3.cx(1, 2)
        w3.ry(np.pi / 4, 1)
        w3.cx(2, 1)
        w3.ry(-1 * np.pi / 4, 1)
        w3.x(1)
        w3.x(2)
        w3.cx(1, 3)
        w3.cx(2, 3)

        # swap test
        w3.h(0)
        for x in range(1, qubits + 1):
            w3.barrier()
            fredkin3(w3, 0, x, x + qubits)
            w3.barrier()
        w3.h(0)

        w3.measure(0, 0)



def repeater_circuit(repeater, choice):

    if choice == 'probabilities':
        repeater.h(0)
        repeater.h(2)
        repeater.cx(0,1)
        repeater.cx(2,3)
        repeater.h(0)
        repeater.h(1)
        repeater.h(2)
        repeater.h(3)
        repeater.cx(1,2)
        repeater.h(3)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(1,2)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(0,1)
        repeater.h(0)
        repeater.h(1)
        repeater.cx(0,1)
        repeater.h(0)
        repeater.h(1)
        repeater.cx(0,1)
        repeater.h(1)
        repeater.cx(1,3)
        repeater.h(1)
        repeater.h(3)
        repeater.cx(0,1)
        repeater.h(0)
        repeater.h(1)
        repeater.cx(0,1)
        repeater.h(0)
        repeater.h(1)
        repeater.cx(0,1)
        for x in range(4):
            repeater.measure(x, x)
    elif choice == 'fidelities':
        qubits = 4
        repeater.h(1)
        repeater.h(3)
        repeater.cx(1, 2)
        repeater.cx(3, 4)
        repeater.h(1)
        repeater.h(2)
        repeater.h(3)
        repeater.h(4)
        repeater.cx(2, 3)
        repeater.h(4)
        repeater.h(2)
        repeater.h(3)
        repeater.cx(2, 3)
        repeater.h(2)
        repeater.h(3)
        repeater.cx(1, 2)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(1, 2)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(1, 2)
        repeater.h(2)
        repeater.cx(2, 4)
        repeater.h(2)
        repeater.h(4)
        repeater.cx(1, 2)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(1, 2)
        repeater.h(1)
        repeater.h(2)
        repeater.cx(1, 2)

        # swap test
        repeater.h(0)
        for x in range(1, qubits + 1):
            repeater.barrier(0)
            fredkin3(repeater, 0, x, x + qubits)
            repeater.barrier(0)
        repeater.h(0)

        repeater.measure(0, 0)


def one_qubit_adder_circuit(qc, choice):

    if choice == 'probabilities':
        qc.ccx(3, 2, 0)
        qc.ccx(4, 3, 0)
        qc.ccx(4, 2, 0)
        qc.cx(4, 1)
        qc.cx(3, 1)
        qc.cx(2, 1)
        for x in range(5):
            qc.measure(x, x)
    elif choice == 'fidelities':
        qubits = 5
        qc.ccx(4, 3, 1)
        qc.ccx(5, 4, 1)
        qc.ccx(5, 3, 1)
        qc.cx(5, 2)
        qc.cx(4, 2)
        qc.cx(3, 2)

        # swap test
        qc.h(0)
        for x in range(1, qubits + 1):
            qc.barrier(0)
            fredkin3(qc, 0, x, x + qubits)
            qc.barrier(0)
        qc.h(0)

        qc.measure(0, 0)


def sort_counts(counts, qubits):
    # Find probabilities

    vec = []
    for x in range(2 ** qubits):
        bin = np.binary_repr(x).zfill(qubits)
        if bin in counts.keys():
            val = counts[bin] / 1024
            vec.append(val)
        else:
            vec.append(0.0)
    return vec


def gate_error_noise_model(dev_name):
    # regular noise model for the backend
    device = provider.get_backend(dev_name)
    properties = device.properties()
    gate_lengths = noise.device.parameters.gate_length_values(properties)
    noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
    basis_gates = noise_model.basis_gates
    coupling_map = device.configuration().coupling_map
    return device, noise_model, basis_gates, coupling_map


def write_data(save_path, vectors):
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for x in vectors:
            writer.writerow(x)
    csvFile.close()

# lists needed for data gathering
circs, circ_qubits = ['teleport', 'wstate', 'ghz', 'repeater', 'adder', 'deutsch'], [3, 3, 4, 4, 5, 5]
circ_algs = [teleportation_circuit, wstate_circuit, GHZ_circuit, repeater_circuit, one_qubit_adder_circuit, deutsch_jozsa_circuit]


def probabilities_test_data(pop):

    # initialize chosen circuit, find variables and noise model parameters
    circ = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder, deutsch)'))
    qubits = int(circ_qubits[circs.index(circ)])
    dev_name = 'ibmq_16_melbourne' # put in choice
    device, noise_model, basis_gates, coupling_map = gate_error_noise_model(dev_name)
    vectors = []

    for x in range(pop):
        # obtain a noisy counts vector
        algorithm = circ_algs[circs.index(circ)] # CHANGE BACK ALL SHOTS
        circuit = initialize_circuit(qubits)
        algorithm(circuit, 'probabilities')
        result = execute(circuit, backend=simulator, shots=1024, noise_model=noise_model,
                         basis_gates=basis_gates, optimization_level=0).result()
        counts = [result.get_counts(i) for i in range(len(result.results))]
        prob_vec = sort_counts(counts[0], qubits)

        # add an unknown class (necessary for tensorflow to take the dataset)
        prob_vec.append(200)
        vectors.append(prob_vec)
        print(prob_vec)

    # store vectors
    save_path = 'ibm_data/' + circ + str(pop) + '_ibm_sim_probabilities_' + dev_name + '.csv'
    write_data(save_path, vectors)


def fidelities_test_data(pop):

    # initialize chosen circuit, find variables and noise model parameters
    circ = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder, deutsch)'))
    qubits = int(circ_qubits[circs.index(circ)])
    dev_name = 'ibmq_16_melbourne'
    device, noise_model, basis_gates, coupling_map = gate_error_noise_model(dev_name)
    vectors = []

    # build size four vectors of chosen initial states which are indexed in get_starting_state()
    for x in range(pop):
        vector = []
        for i in range(4):
            algorithm = circ_algs[circs.index(circ)]
            circuit = initialize_diagnostic_circuit(qubits, i)
            algorithm(circuit, 'fidelities')
            result = execute(circuit, backend=simulator, shots=1024, noise_model=noise_model,
                             basis_gates=basis_gates,optimization_level=0).result()
            counts = [result.get_counts(i) for i in range(len(result.results))]
            # find the p(anc == |0>)
            if '0' in counts[0].keys():
                p_0 = counts[0]['0'] / 1024
                vector.append(p_0)
            else:
                vector.append(0.0)

        # add unknown class (necessary for tensorflow to take the dataset)
        vector.append(200)
        print(vector)
        vectors.append(vector)

    # store vectors
    save_path = 'ibm_data/' + circ + str(pop) + '_ibm_sim_fidelities' + dev_name + '.csv'
    write_data(save_path, vectors)


def main(pop):

    metric_choice = str(input('Which metric? (probabilities, fidelities)'))
    if metric_choice == 'probabilities':
        probabilities_test_data(pop)
    elif metric_choice == 'fidelities':
        fidelities_test_data(pop)


main(100)
