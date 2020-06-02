CommentedCode.py: Qutip code meant to implement a set of randomized circuits to set KNN procedure

GHZ 4 qubit circuit.py: is a qiskit code that visualizes 4 qubit circuit

"permanent training" == Generate KNN training data csv file
GHZPermanentTraining.py:^ for ghz circuit
Teleportation Training.py: ^ teleport circuit
RepeaterPermanentTraining.py: ^ for repeater/entanglement swapping circuit

Hadamard_alteration: qutip code contains the new hadamard error simulation codes that were dispered throughout

QFT(1).py: qiskit code generates a QFT

Quantum Teleportation Code (Fixed) (1).py: qiskit code that obtains fidelity of the teleportation circuit also contains that circuit

SingleVectorClassifier: a qutip code that uses KNN data to classify a single vector rather than evaluate a ML technique

ToleranceTest: qutip code that contains the get_ideal, tolerance, and adjusted KNN functions that account for the tolerance of a real machine, needs as input the fidelity of machine

Verification of Diagnostic Circuit on IBM Quantum Computer: is a qiskit code of the diagnostic circuit (Swap test)

"playground" == a notebook used to figure out how to code
knearestneighbors: trial and error for KNN implementations
qiskitplayground: trial and error for qiskit interface code

KNN-Comparisons
naming convention: (use Corrected-by-for-real-this-time)
2 in front => that all zero inputs are used
Middle => what circuit
number => how many random simulated errors per gate were used
tag at end => what U was used
"new" => new basis vectors
"Had" =>Hadamard
"QFT" => Quantum Fourier Transform
