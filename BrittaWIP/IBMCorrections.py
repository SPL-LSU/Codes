# code to test mitigation methods and more tolerant gates based on machine learning results
# B Manifold 7.14.20
# Things that have to be manually changed: error list in main()
# still someways in progress

import numpy as np
import qiskit as q
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ
from qiskit.providers.aer.noise.device import models
import csv
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
import math
from qiskit.extensions import UnitaryGate

IBMQ.enable_account("5fb2351e186ca52ed62daf9ce94aefaf2992028d9d7d9975df5a501155fe89d7b0e35cf49b1a5137287e24a791dc691f6c4baa08116c8ef2967b86cf3dac533c")
provider = IBMQ.get_provider()
simulator = Aer.get_backend('qasm_simulator')


def ideal_vector_probabilities(circ):
    # classical generated ideal vector

    if circ == 'wstate':
        return  [0.0833, 0.1667, 0.2429, 0.0071, 0.0833, 0.1667, 0.2429, 0.0071]
    elif circ == 'teleport':
        return [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0]
    elif circ == 'ghz':
        return [0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125, 0.0, 0.125]
    elif circ == 'repeater':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
    elif circ == 'adder':
        return [0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313,
                0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313,
                0.0313, 0.0313, 0.0313, 0.0313, 0.0313, 0.0313]

def ideal_vector_fidelities(circ):
    # classical generated ideal vector

    if circ == 'wstate':
        return [0.9134, 0.4516, 0.8088, 0.4082]
    elif circ == 'teleport':
        return [0.4344, 0.4344, 0.4344, 0.4344]
    elif circ == 'ghz':
        return [0.7254, 0.0625, 0.0625, 0.0625]
    elif circ == 'repeater':
        return [0.5313, 0.5313, 0.0625, 0.0625]
    elif circ == 'adder':
        return [1.0, 0.5156, 0.0312, 0.5156]


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
    for x in bin:
        if x == '1':
            qc.x(pos + 1)
            qc.x(pos + qubits + 1)
        pos += 1
    for x in range(0, qubits):
        qc.h(x)
        qc.h(x + qubits)
    return qc


def v_matrix(dagger):

    w = (1 + 1j) / 2
    v = np.array([[w, -1j * w],[-1j * w, w]])
    if dagger:
        return np.conj(v).T
    else:
        return v


def fredkin3(qc, control, t1, t2):
    qc.cx(t2, t1)
    qc.ccx(control, t1, t2)
    qc.cx(t2, t1)
    return


def choose_gate(qc, circuit_errors, gate_type, inds, rotation_spec, theta):
    # choose which version of the gate to implement (regular or decomposed) based on error list
    # dex is index of gate in circuit
    # still experimenting with these

    if gate_type == 'CNOT':
        if gate_type in circuit_errors:
            qc.barrier()
            qc.h(inds[1])
            qc.barrier()
            qc.cz(inds[0], inds[1])
            qc.barrier()
            qc.h(inds[1])
            qc.barrier()
        else:
            qc.cx(inds[0], inds[1])

    elif gate_type == 'X':
        if gate_type in circuit_errors:
            qc.h(inds)
            qc.barrier()
            qc.z(inds)
            qc.barrier()
            qc.h(inds)
        else:
            qc.x(inds)

    elif gate_type == 'HADAMARD':
        if gate_type in circuit_errors:
            qc.x(inds)
            qc.barrier()
            qc.ry(-np.pi / 2)
        else:
            qc.h(inds)

    elif gate_type == "CONTROLZ":
        if gate_type in circuit_errors:
            qc.cz(inds[1], inds[0])
        else:
            qc.cz(inds[0], inds[1])

    elif gate_type == "ROTATION":   # Not implemented Yet
        if gate_type in circuit_errors:
            if rotation_spec == 'X':
                pass
            elif rotation_spec == 'Y':
                pass
            elif rotation_spec == 'Z':
                pass
        else:
            if rotation_spec == 'X':
                qc.rx(theta, inds)
            elif rotation_spec == 'Y':
                qc.ry(theta, inds)
            elif rotation_spec == 'Z':
                qc.rz(theta, inds)

    elif gate_type == 'TOFFOLI':
        if gate_type in circuit_errors:
            c1, c2, t = inds[0], inds[1], inds[2]
            v = UnitaryGate(v_matrix(False)).control()
            v_dagger = UnitaryGate(v_matrix(True)).control()
            qc.append(v, [c2, t])
            qc.barrier()
            qc.h(c2)
            qc.barrier()
            qc.cz(c1, c2)
            qc.barrier()
            qc.h(c2)
            qc.barrier()
            qc.append(v_dagger, [c2, t])
            qc.barrier()
            qc.h(c2)
            qc.barrier()
            qc.cz(c1, c2)
            qc.barrier()
            qc.h(c2)
            qc.barrier()
            qc.append(v, [c1, t])
        else:
            qc.ccx(inds[0], inds[1], inds[2])


def GHZ_circuit(qc, circuit_errors, metric_choice):

    if metric_choice == 'probabilities':
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        for x in range(4):
            qc.measure(x, x)
    elif metric_choice == 'fidelities':
        qubits = 4
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (3, 4), None, None)
        for x in range(1, qubits + 1):
            fredkin3(qc, 0, x, x + qubits)
        qc.h(0)
        qc.measure(0, 0)


def teleportation_circuit(qc, circuit_errors, metric_choice):

    if metric_choice == 'probabilities':
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 2), None, None)
        for x in range(3):
            qc.measure(x, x)
    elif metric_choice == 'fidelities':
        qubits = 3
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 3), None, None)
        for x in range(1, qubits + 1):
            fredkin3(qc, 0, x, x + qubits)
        qc.h(0)
        qc.measure(0, 0)

# somewhere an ry dex didn't get updated...
def wstate_circuit(qc, circuit_errors, metric_choice):

    if metric_choice == 'probabilities':
        choose_gate(qc, circuit_errors, 'ROTATION', 0, 'Y', -1.23096)
        choose_gate(qc, circuit_errors, 'X', 1, None, None)
        choose_gate(qc, circuit_errors, 'X', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'ROTATION', 0, 'Y', np.pi / 4)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 0), None, None)
        choose_gate(qc, circuit_errors, 'ROTATION', 0, 'Y', -1 * np.pi / 4 )
        choose_gate(qc, circuit_errors, 'X', 0, None, None)
        choose_gate(qc, circuit_errors, 'X', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        for x in range(3):
            qc.measure(x, x)
    elif metric_choice == 'fidelities':
        qubits = 3
        choose_gate(qc, circuit_errors, 'ROTATION', 1, 'Y', -1.23096)
        choose_gate(qc, circuit_errors, 'X', 2, None, None)
        choose_gate(qc, circuit_errors, 'X', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'ROTATION', 1, 'Y', np.pi / 4)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 1), None, None)
        choose_gate(qc, circuit_errors, 'ROTATION', 1, 'Y', -1 * np.pi / 4 )
        choose_gate(qc, circuit_errors, 'X', 1, None, None)
        choose_gate(qc, circuit_errors, 'X', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 3), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        for x in range(1, qubits + 1):
            fredkin3(qc, 0, x, x + qubits)
        qc.h(0)
        qc.measure(0, 0)


def repeater_circuit(qc, circuit_errors, metric_choice):

    if metric_choice == 'probabilities':
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 3), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 0, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (0, 1), None, None)
        for x in range(4):
            qc.measure(x, x)
    elif metric_choice == 'fidelities':
        qubits = 4
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (3, 4), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 4, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 4, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 3, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 3), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 4, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 1, None, None)
        choose_gate(qc, circuit_errors, 'HADAMARD', 2, None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (1, 2), None, None)
        for x in range(1, qubits + 1):
            fredkin3(qc, 0, x, x + qubits)
        qc.h(0)
        qc.measure(0, 0)


def one_qubit_adder_circuit(qc, circuit_errors, metric_choice):

    if metric_choice == 'probabilities':
        choose_gate(qc, circuit_errors, 'TOFFOLI', (3, 2, 0), None, None)
        choose_gate(qc, circuit_errors, 'TOFFOLI', (4, 3, 0), None, None)
        choose_gate(qc, circuit_errors, 'TOFFOLI', (4, 2, 0), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (4, 1), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (3, 1), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (2, 1), None, None)
        for x in range(5):
            qc.measure(x, x)
    elif metric_choice == 'fidelities':
        qubits = 5
        choose_gate(qc, circuit_errors, 'TOFFOLI', (4, 3, 1), None, None)
        choose_gate(qc, circuit_errors, 'TOFFOLI', (5, 4, 1), None, None)
        choose_gate(qc, circuit_errors, 'TOFFOLI', (5, 3, 1), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (5, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (4, 2), None, None)
        choose_gate(qc, circuit_errors, 'CNOT', (3, 1), None, None)
        for x in range(1, qubits + 1):
            fredkin3(qc, 0, x, x + qubits)
        qc.h(0)
        qc.measure(0, 0)


def sort_counts(counts, qubits):

    vec = []
    for x in range(2 ** qubits):
        bin = np.binary_repr(x).zfill(qubits)
        if bin in counts.keys():
            val = counts[bin] / 1024
            vec.append(val)
        else:
            vec.append(0.0)

    return vec


def trace_distance(vec1, vec2):

    distance = 0
    n = len(vec1)
    for x in range(n):
        distance += abs(vec1[x] - vec2[x])
    return distance / 2


def euclidean_distance(vec1, vec2):
    dis = 0
    length = len(vec1)
    for x in range(length):
        dis += pow((vec1[x] - vec2[x]), 2)
    return math.sqrt(dis)


def run_and_sort(noise_model, qubits, circuit):

    result = execute(circuit, backend=simulator, shots=1024, noise_model=noise_model, optimization_level=0).result()
    counts = [result.get_counts(i) for i in range(len(result.results))]
    prob_vec = sort_counts(counts[0], qubits)

    return prob_vec


def test(pop):

    circs, circ_qubits = ['teleport', 'wstate', 'ghz', 'repeater', 'adder'], [3, 3, 4, 4, 5]
    circ_algs = [teleportation_circuit, wstate_circuit, GHZ_circuit, repeater_circuit, one_qubit_adder_circuit]
    metric_choice = str(input('Which metric? (probabilities, fidelities)'))

    circ = str(input('Which circuit? (teleport, wstate, ghz, repeater, adder)'))
    qubits = int(circ_qubits[circs.index(circ)])
    circuit_errors = ['CNOT'] # Has to be manually changed based on ML output
    ideal, dev_name = [], ''

    if metric_choice == 'probabilities':
        ideal = ideal_vector_probabilities(circ)
        dev_name += 'ibmq_london'
    elif metric_choice == 'fidelities':
        ideal = ideal_vector_fidelities(circ)
        dev_name += 'ibmq_16_melbourne'
    print(dev_name)
    device = provider.get_backend(dev_name)
    properties = device.properties()
    # Only modeling gate errors
    errors = models.basic_device_gate_errors(properties, gate_error=True, thermal_relaxation=False)
    noise_model = NoiseModel()
    for name, qubitz, error in errors:
        noise_model.add_quantum_error(error, name, qubitz)

    distances, raw_distances = 0, 0
    for x in range(pop):

        algorithm = circ_algs[circs.index(circ)]
        if metric_choice == 'probabilities':
            # Including altered gates
            circuit = initialize_circuit(qubits)
            algorithm(circuit, circuit_errors, metric_choice)
            prob_vec = run_and_sort(noise_model, qubits, circuit)
            print("Altered:", prob_vec)
            # No altered gates
            circuit_raw = initialize_circuit(qubits)
            algorithm(circuit_raw, [], metric_choice)
            prob_vec_raw = run_and_sort(noise_model, qubits, circuit_raw)
            print("Unaltered:", prob_vec_raw)

            raw_distances += trace_distance(prob_vec_raw, ideal)
            distances += trace_distance(prob_vec, ideal)

        elif metric_choice == 'fidelities':
            # Including Altered gates
            vector_alt, vector_raw = [], []
            for i in range(4):
                circuit = initialize_diagnostic_circuit(qubits, i)
                algorithm(circuit, circuit_errors, metric_choice)
                prob_vec = run_and_sort(noise_model, 1, circuit)
                vector_alt.append(prob_vec[0])
            print(vector_alt)
            # No altered gates
            for i in range(4):
                circuit_raw = initialize_diagnostic_circuit(qubits, i)
                algorithm(circuit_raw, [], metric_choice)
                prob_vec_raw = run_and_sort(noise_model, 1, circuit_raw)
                vector_raw.append(prob_vec_raw[0])
            print(vector_raw)

            distances += euclidean_distance(vector_alt, ideal)
            raw_distances += euclidean_distance(vector_raw, ideal)

    print("Average distance for alternate circuit:", distances / pop)
    print("Average distance for original circuit:", raw_distances / pop)


test(5)


# More than one backend matches the criteria -- ?
