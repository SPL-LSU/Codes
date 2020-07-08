# code to gather noisy simulator or hardware test data for teleportation circuit

# fredkins gates: https://pdfs.semanticscholar.org/4774/7792a7b13028e47c9daa2259f77d264a27a8.pdf

import numpy as np
import qiskit as q
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ
import csv
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
import math
IBMQ.enable_account("cfd254684a5e9ca4eb51ee411ca502e2be9d07a4755591a3a6ce7e773a95f442347667331621b0c1844ff26dd4536c99f05608c0ad79812938d710c69919bdb4")
provider = IBMQ.get_provider()
simulator = Aer.get_backend('qasm_simulator')
dev_name='ibmq_16_melbourne'

def teleportState(qc):

    qc.h(2)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.h(1)
    qc.cx(2, 3)
    qc.cz(1, 3)
    return

def create_basic_states(choice):

    qc = QuantumCircuit(7, 1)
    bin = np.binary_repr(choice).zfill(3)
    dex = 0
    for i in bin:
        if i == '1':
        # ancilla is q0
            qc.x(dex + 1)
            qc.x(dex + 4)
        dex += 1

    return qc


def hadamard_state(choice):

    # going off readout of statevectors from classical code
    # where the basic states before hadamard unitary are 001, 011, 101, 111
    qc = create_basic_states(choice)
    for i in range(3):
        qc.h(1 + i)
        qc.h(4 + i)

    return qc


def fourier_state(choice):

    qc = create_basic_states(choice)
    for i in range(0, 3):
        qc.h(1 + i)
        for j in range(i + 1, 3):
            qc.cu1(math.pi / (2 ** j), j + 1, i + 1)
        qc.h(4 + i)
        for j in range(i + 1, 3):
            qc.cu1(math.pi / (2 ** j), j + 4, i + 4)

    return qc


def v_matrix(dagger):

    w = (1 + 1j) / 2
    v = np.array([[w, -1j * w],[-1j * w, w]])
    if dagger:
        return np.conj(v).T
    else:
        return v

def fredkin(qc, control, t1, t2):

    # assumes t1 < t2 since ancilla is q0
    v = qeu.UnitaryGate(v_matrix(False)).control()
    v_dagger = qeu.UnitaryGate(v_matrix(True)).control()
    qc.cx(control, t1)
    qc.append(v, [t1, 0])
    qc.append(v, [t2, 0])
    qc.cx(t2, t1)
    qc.append(v_dagger, [t1, 0])
    qc.cx(t2, t1)
    qc.cx(control, t1)

    return


def teleport_test_data(pop):

    algorithms = [create_basic_states, hadamard_state, fourier_state]
    paths = ['Teleport_Melbourne_Basic100.csv', 'Teleport_Melbourne_Had100.csv', 'Teleport_Melbourne_QFT100.csv']

    device = provider.get_backend(dev_name)
    properties = device.properties()
    gate_lengths = noise.device.parameters.gate_length_values(properties)
    noise_model = noise.device.basic_device_noise_model(properties, gate_lengths=gate_lengths)
    basis_gates = noise_model.basis_gates
    coupling_map = device.configuration().coupling_map

    i = 0
    for alg in algorithms:
        save_path = paths[i]
        vectors = []

        for j in range(pop):
            prob_vec = []
            for choice in [1, 3, 5, 7]:
                qc = alg(choice)
                teleportState(qc)
                for x in range(3):
                    fredkin(qc, 0, x + 1, x + 4)
                qc.measure(0, 0)

                result = execute(qc, backend=simulator, shots=1000, noise_model=noise_model,
                                 basis_gates=basis_gates, coupling_map=coupling_map).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                if '0' in counts[0].keys():
                    prob = counts[0]['0'] / 1000
                    prob_vec.append(prob)
            vectors.append(prob_vec)
        for v in vectors:
            print(v)

        with open(save_path,'a',newline='') as csvFile:
            writer = csv.writer(csvFile)
            for x in vectors:
                writer.writerow(x)
        csvFile.close()
        i += 1

    return

teleport_test_data(100)























