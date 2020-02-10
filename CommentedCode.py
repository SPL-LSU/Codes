import numpy as np
import qutip as qt
import math
import operator
from itertools import product
from random import randint, uniform, choice,random
from qutip.qip.algorithms import qft
import time

#splits the data set into testing and training data. Need to initialize empty vectors first
def handleDataset(array,split,trainingSet=[],testSet=[]):
    #with open(filename,'r') as csvfile:
        #lines = csv.reader(csvfile)
        #dataset=list(lines)
    dataset=array
    for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y]=float(dataset[x][y])
            if random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        #print(trainingSet, 'aaaahhh', testSet)
    return 0

#Finds the euclidean distance bewteen two vectors of length 'length;
def euclideanDistance(ins1,ins2,length):
    dis=0
    for x in range(length):
        dis += pow((ins1[x]-ins2[x]),2)
    return math.sqrt(dis)

#finds the k points nearest a test point
def getKNeighbors(trainingSet,test,k):
    distances=[]
    length=len(test)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(test,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#determines the classes of a vector of neighbors and returns a prediction of a test point's class
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
#Finds the accuracy of a test set prediction   
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        #if testSet[x][-1] != predictions[x]:
            #print("predicted:" + predictions[x],"actual:" + testSet[x][-1])
    return (correct/float(len(testSet)))*100.0
 
#Performs KNN classification given a dataset, a training-test split, and k
def KNN(path,split,k):
    testingset=[]
    trainingset=[]
    
    #Insert csv file name and split here:
    #path=input("Give csv file location, remembering to use forward slashes: ")
    split = split
    split=float(split)
    if split > 1 or split < 0:
        print("Incorrect split input given. Default used.")
        split = 0.66
    handleDataset(path,split,trainingset,testingset)
    
    # generate predictions 
    predictions=[] 
    k=k
    for x in range(len(testingset)): 
        neighbors = getKNeighbors(trainingset,testingset[x],k) 
        result = getResponse(neighbors) 
        predictions.append(result) 
        #print('> predicted=' + repr(result) + ',actual=' + repr(testingset[x][-1]))
        
        
        
    accuracy = getAccuracy(testingset,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


###############################################################################
#The thing we've been hard-coding everywhere to fix dimension problems
def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

#2D identity that we needed to define globally I don't remember why
id2=qt.Qobj(np.identity(2))

#Applies an array of gates in a circuit and applies them to a state
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

#Takes the "distance" (Fidelity metric) between the input & reference state    
def dis(state1,state2):
    (n,m)=state1.shape
    if state1.type is not 'ket' or state2.type is not 'ket':
        print("Error: One or more input to the distance function is not a ket." )
        p_0=0
    else:
        fid=qt.fidelity(state1,state2)
        p_0 = 1/n + (n-1)*fid/n
    return p_0

#messed up hadamard Allee coded most of this
def multi_qubit_hadamard(regular_hadamard_gate):
    theta = uniform(0.0,math.pi*2.0)
    (n,n) = regular_hadamard_gate.shape
    N = np.int(((np.log(n))/(np.log(2))))
    phase=qt.globalphase(theta,N)
    phase=tensor_fix(phase)
    reg=tensor_fix(regular_hadamard_gate)
    multi_qubit_hadamard = phase*reg
    multi_qubit_hadamard = tensor_fix(multi_qubit_hadamard)    
    return multi_qubit_hadamard

#She's figuring out the composition of the Hadamard
def identifying_identities_and_hadamards(multi_qubit_hadamard):
    (n,n) = multi_qubit_hadamard.shape
    Q = np.log(n)/np.log(2)#Q is the number of qubits
    multi_qubit_hadamard = multi_qubit_hadamard.full()
    m = multi_qubit_hadamard.item(0) #m is the numberical value of cell from original array
    m = m.real
    H = np.log(m)/np.log(1/math.sqrt(2)) #H is the number of hadamrd gates applied
    H = np.rint(H) # making H an integer incase of slight rounding errors
    return Q,H

#She's listing every decomposition possible of said Hadamard
def combination_possibilities(Q,H):
    all_possible = list(product('HI',repeat = int(Q)))
    x=0
    limited_possibilities = []
    for ele in all_possible:
        h = count_occurances(ele,'H')
        h = int(h)
        if h == H:
            limited_possibilities.append(ele)
        else:
            x = x+1 #this else is set up to that it can be used to debug if necessary
    return limited_possibilities  

#Honestly not sure
def count_occurances(ele, gate):
    count = 0
    for x in ele:
        if (x == gate):
            count = count +1
    return count

#Recombination part methinks?
def tensor_elements_correctly(ele):
    if ele[0] == 'H':
        possible_array = qt.hadamard_transform(1)
    elif ele[0] == 'I':
        possible_array = qt.identity(2)
        
    for x in ele[1:len(ele)]:
        if x == 'H':
            possible_array = tensor_fix(qt.tensor(possible_array,qt.hadamard_transform(1)))
        elif x == 'I':
            possible_array = tensor_fix(qt.tensor(possible_array,qt.identity(2)))
    return possible_array,ele
    
#????
def combinations_to_arrays(limited_possibilities):
    possible_arrays = []
    array_of_combinations= []
    for ele in limited_possibilities:
        possible_array,ele = tensor_elements_correctly(ele)
        possible_arrays.append(possible_array)
        array_of_combinations.append(ele)
    return possible_arrays,array_of_combinations

def correct_combination(multi_qubit_hadamard,possible_arrays,array_of_combinations):
    array_location = possible_arrays.index(multi_qubit_hadamard)
    combination = array_of_combinations[array_location]
    return combination

#This is the part where she adds relative phase shift
def hadamard_with_phase_shift():
    theta = uniform(0.0,math.pi*2.0)
    shifted_hadamard = qt.globalphase(theta,1)*(qt.hadamard_transform(1))
    return qt.Qobj(shifted_hadamard)

#Generating the bad hadamard?
def tensor_elements_incorrectly(combination):
    if combination[0] == 'H':
        altered_array = hadamard_with_phase_shift()
    elif combination[0] == 'I':
        altered_array = qt.identity(2)
        
    for x in combination[1:len(combination)]:
        if x == 'H':
            altered_array = tensor_fix(qt.tensor(altered_array,hadamard_with_phase_shift()))
        elif x == 'I':
            altered_array = tensor_fix(qt.tensor(altered_array,qt.identity(2)))
    altered_array= altered_array.full()
    altered_array = altered_array.real
    altered_array = qt.Qobj(altered_array)
    return altered_array
        
def alter_multi_qubit_hadamard_specifically(multi_qubit_hadamard):
    Q,H = identifying_identities_and_hadamards(multi_qubit_hadamard)
    limited_possibilities = combination_possibilities(Q,H)
    possible_arrays,array_of_combinations = combinations_to_arrays(limited_possibilities)
    combination = correct_combination(multi_qubit_hadamard,possible_arrays,array_of_combinations)
    altered_hadamard = tensor_elements_incorrectly(combination)
    return altered_hadamard

#code which gives an original unitary gate
def random_unitary_gate(delta,alpha,theta,beta,value):
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
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

#code which takes an angle and alters the gate
def random_altered_unitary_gate(delta,alpha,theta,beta,value):
    if delta == 0.0 and alpha == 0.0 and theta == math.pi and value == True:
        angles = ['delta','alpha','beta']
    else:
        angles = ['delta','alpha','theta','beta']
    altered_variable = choice(angles)
    if altered_variable == 'delta':
        delta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'alpha':
        alpha = uniform(0.0,2.0*math.pi)
    if altered_variable == 'theta':
        theta = uniform(0.0,2.0*math.pi)
    if altered_variable == 'beta':
        beta = uniform(0.0,2.0*math.pi)
    gate = qt.Qobj(qt.phasegate(delta)*qt.rz(alpha)*qt.ry(theta)*qt.rz(beta))
    if value == True:
        gate = gate *qt.Qobj([[0,1],[1,0]])
    else:
        gate = gate
    return gate
   
#gives both an original and altered unitary gate
#can be commented to return original gate, corresponding altered gate(only one thing different from original), or both
def unitary_gate(choice):
    delta,alpha,theta,beta,value = random_angles()
    original = random_unitary_gate(delta,alpha,theta,beta,value)
    matrix=original
    if choice:
        altered = random_altered_unitary_gate(delta,alpha,theta,beta,value)
        matrix=altered
    return (matrix,[delta,alpha,theta,beta,value])

#Gives an altered cnot
def rot(qubits,choice):
    a=randint(0,qubits-2)
    b=randint(a+1,qubits-1)
    k=qt.cnot(qubits,a,b)
    k=k.full()
    if choice:
        k=np.random.permutation(k)
        k=qt.Qobj(k)
        #k=k*k.dag()
        k=tensor_fix(k)
    cn_final=qt.Qobj(k)
    return cn_final

#Generates a random circuit
def arb_circuit_generator(length,qubits):
    circuit=[]
    angles=[]
    controls=[]
    id2=np.identity(2)
    id2=qt.Qobj(id2)
    n=2**qubits
    Had = 0
    CNOT = 0
    Ran = 0
    while len(circuit)<2*length:
        seed=randint(1,3)
        if seed ==1:
            temp=qt.hadamard_transform()
            temp=tensor_fix(temp)
            circuit.append(temp)
            circuit.append("Hadamard")
            Had = Had +1
        if seed == 2:
            temp=rot(qubits,False)
            circuit.append(temp)
            circuit.append("CNOT")
            CNOT = CNOT +1
        if seed == 3:
            (temp,ang)=unitary_gate(False) 
            circuit.append(temp)
            circuit.append("Random Unitary")
            angles.append(ang)
            Ran = Ran +1
    for i in circuit:
        if type(i) == str:
            continue
        else:
            i=tensor_fix(i)
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        elif circuit[i].full().shape == (n,n):
            continue
        else:
            place=randint(1,qubits)
            before=qubits-place
            after=qubits-before-1
            temp=circuit[i]
            temp=qt.Qobj(temp)
            if place == 1:
                circuit[i]=qt.tensor(temp,id2)
            else:
                while before > 0:
                    if circuit[i].full().shape == (n,n):
                        break
                    temp=qt.tensor(id2,temp)
                    temp=tensor_fix(temp)
                    circuit[i]=temp
                    before=before-1
                while after > 0:
                    if circuit[i].full().shape == (n,n):
                        break
                    temp=qt.tensor(temp,id2)
                    temp=tensor_fix(temp)
                    circuit[i]=temp
                    after=after-1
    composition = ["Hadamards:",Had,"CNOT:",CNOT,"Random Unitary:",Ran]
    return (circuit,angles,controls,composition)

#Creates the list of gates in the circuit
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
            if circuit[i+1] == "Hadamard":
                it_h+=1
                t=str(it_h)
                t="Hadamard" + t
                cat.append(t)
            elif circuit[i+1] == "CNOT":
                it_not+=1
                t=str(it_not)
                t="CNOT" + t
                cat.append(t)
            elif circuit[i+1] == "Random Unitary":
                it_rand+=1
                t=str(it_rand)
                t="Random Unitary" + t
                cat.append(t)
        elif circuit[i+1] == "Identity":
            it_id +=1
            t=str(it_id)
            t="Measurement" + t
            cat.append(t)
        else:
            cat.append(circuit[i+1])
        i+=2
    return(cat)
    
#Generates the input qubit vectors
def gen_basis_vectors(n,dims,choice):
    vectors=[]
    fock_states=[]
    bits=int(math.log(n,2))
    for i in range(n):
        state=qt.basis(n,i)
        fock_states.append(state)
    test_opt1=[fock_states[0],fock_states[0]+fock_states[1],fock_states[-1]+fock_states[0],fock_states[-2]+fock_states[1]]
    test_opt2=[fock_states[0],fock_states[0]+fock_states[1],fock_states[-1]+fock_states[0]+fock_states[1],fock_states[-2]+fock_states[1]+fock_states[0]+fock_states[-1]]
    if choice == 1: #Basis states
        vectors = fock_states
        q=qt.Qobj(np.ones(n))
        q=q.unit()
        vectors.append(q)
        vectors.append(state) #there is no two for now
    elif choice == 3: #Hadamard option 1
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in test_opt1:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 4: #QFT option 1
        quft=tensor_fix(qft.qft(bits))
        for state in test_opt1:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 5: #Hadamard option 2
        h=tensor_fix(qt.hadamard_transform(bits))
        for state in test_opt2:
            state_n=h*state
            state_n=state_n.unit()
            vectors.append(state_n)
    elif choice == 6: #QFT option 2
        quft=tensor_fix(qft.qft(bits))
        for state in test_opt2:
            state_n=quft*state
            state_n=state_n.unit()
            vectors.append(state_n)
        vectors.append(state_n)
    return vectors

################################################################################

#Used for debugging if the name didn't give it away
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
        #else:
            #print("fixed it!")  #there were too many "fixed it!" messages
    return gate
        
#This is what implements Allee's code
def h_reassign(hada):
    seed = randint(0,1)
    if seed == 0:
        alt_had=multi_qubit_hadamard(hada)
    if seed == 1:
        alt_had=alter_multi_qubit_hadamard_specifically(hada)
    return alt_had

#Takes as input a circuit w/no str, a set of angles, state and basis vectors, categories, a population
#As I was improvising a lot, the Colin Mochrie name STAYS
def colin_mochrie(circuit,angles,vectors,basis,pop,cat,qubits): 
    probabilities=[]
    n=2**qubits
    index=0
    for j in range(pop):
        state1=basis[0]
        state2=basis[1]
        state3=basis[2]
        state4=basis[3]
        comp1=vectors[0]
        comp2=vectors[1]
        comp3=vectors[2]
        comp4=vectors[3]
        for i in range(len(circuit)):
            gate_holder=circuit[i]
            name=cat[i]
            if "Hadamard" in name:
                alt_gate=h_reassign(gate_holder)
                #print(name,alt_gate)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(h_reassign(circuit[i]),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            if "Random" in name:
                (delta,alpha,theta,beta,value)=angles[index]
                alt_gate=random_altered_unitary_gate(delta,alpha,theta,beta,value)
                #print(name,alt_gate)
                alt_gate=gate_troubleshooter(alt_gate,n)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(random_altered_unitary_gate(delta,alpha,theta,beta,value),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            if "CNOT" in name:
                alt_gate=rot(qubits,True)
                alt_gate=gate_troubleshooter(alt_gate,n)
                #print(name,alt_gate)
                circuit[i]=qt.Qobj(alt_gate)
                count=0
                while circuit[i] == gate_holder:
                    circuit[i]=gate_troubleshooter(rot(qubits,True),n)
                    print("fixing...")
                    count+=1
                    if count == 20:
                        break
            final1=basic_b(state1,circuit)
            prob1=dis(final1,comp1)
            final2=basic_b(state2,circuit)
            prob2=dis(final2,comp2)
            final3=basic_b(state3,circuit)
            prob3=dis(final3,comp3)
            final4=basic_b(state4,circuit)
            prob4=dis(final4,comp4)
            probabilities.append([prob1,prob2,prob3,prob4,name])
            count=0
            while circuit[i] != gate_holder:
                circuit[i]=gate_holder
                count+=1
                if count == 10:
                    break
    return probabilities

def main():
    #Asks console for many convenient inputs
    pop=input("How many of each gate do you want to populate? ")
    pop=int(pop)
    qubits=input("How many qubits? ")
    qubits=int(qubits)
    length=input("how many gates in the circuit? ")
    length=int(length)
    split=input("Give training set split, in the form of a number between 0 and 1: ") #0.8 is typical
    k =input("Give a k value: ")
    k=int(k)
    (circuit,angles,controls,composition)=arb_circuit_generator(length,qubits) #generates a rando circuit with all its acutrements
    n=2**qubits
    cat=categorize(circuit) #Creates the list of classifications used in KNN
	#Tells ya what's in the circuit (convenient for random circuits)
    print(cat) 
    print(composition)
    alt=[]
    for i in range(len(circuit)):
        if type(circuit[i]) == str:
            continue
        else:
            alt.append(gate_troubleshooter(circuit[i],n))
			#This is all the fun data taking parts where I compare different U's & V's
    choice = [1,3,4,5,6]
    vector_name = ['Basis','Hadamard 1','QFT','Hadamard 2','Fourier State']
    index = 0
    for x in choice:
        choice = x
        vectors=gen_basis_vectors(n,n,choice)
        basis=gen_basis_vectors(n,n,choice)
        print(vector_name[index])
        index = index+1
        probs=colin_mochrie(alt,angles,vectors,basis,pop,cat,qubits)
        KNN(probs,split,k)
    return 0

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time)) 
