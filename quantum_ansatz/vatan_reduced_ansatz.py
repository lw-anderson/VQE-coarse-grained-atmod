# Original code: (C) Copyright Martin Kiffner
# Modified by Lewis Anderson
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister

from quantum_ansatz.quantum_ansatz import QuantumAnsatz


class VatanReducedAnsatz(QuantumAnsatz):
    """ 
    Modified version of VatanAnsatz where the blocks have been reduced from 6 to 4 variational parameters. Blocks no
    longer implement arbitrary SO(4) operation.
    """

    def __init__(self, num_qubits, depth, cyclic=False, interact_first=False, offset=1, coupling_range=1, flag=True):

        super().__init__(num_qubits, depth)

        # flag = True means that even depth layers apply 2 qubit gate between first and last qubits
        if not type(cyclic) == bool:
            raise ValueError("Cyclic must be bool")

        # flag = True means that the layers of gates start with blocks that connect second with third, fourth with fifth
        # etc. qubits. In the case of two qubits per oscillator, this corresponds to starting with inter-oscillator
        # interactions.
        if not type(interact_first) == bool:
            raise ValueError("interact_first must be bool.")

        # flag = True means state preparation, false is inverse
        if not type(flag) == bool:
            raise ValueError("Flag must be bool.")

        if offset not in [0, 1] or coupling_range not in [1, 2]:
            raise ValueError("Invalid offset or coupling range.")

        self.interact_first = interact_first
        self.coupling_layer_offset = offset
        self.coupling_range = coupling_range
        self.cyclic = cyclic
        self.flag = flag

        # create symbolic parameters

        self.num_a = self.num_qubits // 2

        if cyclic:
            self.num_b = self.num_qubits // 2
        else:
            self.num_b = (self.num_qubits - 1) // 2

        self.params_per_block = 4
        self.params_per_start_block = 4

        if not interact_first:

            self.num_a_cols = (self.depth // 2) + (self.depth % 2)
            self.num_b_cols = self.depth // 2

            num_params = self.params_per_start_block * self.num_a + self.params_per_block * (
                    (self.num_a_cols - 1) * self.num_a + self.num_b_cols * self.num_b)

        else:

            self.num_a_cols = (self.depth // 2)
            self.num_b_cols = self.depth // 2 + (self.depth % 2)
            num_params = self.params_per_block * (self.num_a_cols * self.num_a + self.num_b_cols * self.num_b)

        params = [Parameter("p" + n.__str__()) for n in range(num_params)]
        self.params = params

    def _create_circuit(self):
        """
        Calculate a circuit that creates the ansatz.
        """

        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q, name='QuantumAnsatz')

        # Start off with the first column
        ind = 0

        if not self.interact_first:
            for i in range(self.num_a):
                hvar = self.get_operator(ind)
                ind += 1
                circuit.append(hvar, [q[2 * i], q[2 * i + 1]])
            remaining_layers = range(1, self.depth)
        else:  # skip over first layer of intra oscillator gates
            remaining_layers = range(1, self.depth + 1)

        for k in remaining_layers:
            if k % 2:
                coupling_qubit_order = list(range(self.num_qubits))
                if self.coupling_range == 2:
                    for i in range(self.num_qubits // 2):
                        if not (i + self.coupling_layer_offset) % 2:
                            coupling_qubit_order[2 * i], coupling_qubit_order[2 * i + 1] \
                                = coupling_qubit_order[2 * i + 1], coupling_qubit_order[2 * i]
                for i in range(self.num_b):
                    hvar = self.get_operator(ind)
                    ind += 1
                    ctrl = coupling_qubit_order[(2 * i + 1) % self.num_qubits]
                    tgt = coupling_qubit_order[(2 * i + 2) % self.num_qubits]
                    circuit.append(hvar, [q[ctrl], q[tgt]])
            else:
                for i in range(self.num_a):
                    hvar = self.get_operator(ind)
                    ind += 1
                    circuit.append(hvar, [q[2 * i], q[2 * i + 1]])

        # consistency check
        if not self.interact_first:
            assert (ind - self.num_a) * self.params_per_block + self.num_a * self.params_per_start_block == len(
                self.params), \
                ValueError("Something went wrong!")
        else:
            assert ind * self.params_per_block == len(self.params), \
                ValueError("Something went wrong!")

        if not self.flag:
            circuit = circuit.inverse()

        return circuit.to_instruction()

    def get_operator(self, ind):

        local_params = self.get_local_params(ind)

        ct = QuantumCircuit(2)
        # qr = ct.qregs[0]
        ct.rz(-np.pi / 2, 0)
        ct.rz(-np.pi / 2, 1)
        ct.ry(-np.pi / 2, 1)
        ct.cx(1, 0)

        ct.rz(local_params[0], 0)
        ct.ry(local_params[1], 0)

        ct.rz(local_params[2], 1)
        ct.ry(local_params[3], 1)

        ct.cx(1, 0)
        ct.ry(np.pi / 2, 1)
        ct.rz(np.pi / 2, 0)
        ct.rz(np.pi / 2, 1)

        return ct.to_instruction()

    def get_local_params(self, ind):
        if ind < self.num_a and not self.interact_first:
            local_params = self.params[self.params_per_block * ind:self.params_per_block * (ind + 1)]
        else:
            if not self.interact_first:
                ind2 = ind - self.num_a
                offset = self.params_per_block * self.num_a
            else:
                ind2 = ind
                offset = 0
            local_params = self.params[
                           (offset + self.params_per_block * ind2):(offset + self.params_per_block * (ind2 + 1))]
        return local_params

    def get_initial(self, initial_state='equal'):
        """
        Calculate the initial state corresponding to a constant function.
        """
        self.circuit  # create circuit to populate self.params

        if initial_state == 'equal':
            vec = [np.pi, 0.5 * np.pi, -0.5 * np.pi] * self.num_a
            if self.num_b_cols > 0:
                if self.num_qubits % 2:
                    hv = [0, 0, 0, 0, 0, 0] * (self.num_b - 1) + [0, 0, 0, 0, 0, 0.5 * np.pi]
                else:
                    hv = [0, 0, 0, 0, 0, 0] * self.num_b
            else:
                hv = []

            initial_values = vec + hv + [0, 0, 0, 0, 0, 0] * (
                    (self.num_a_cols - 1) * self.num_a + (self.num_b_cols - 1) * self.num_b)

        elif initial_state == 'zero':
            initial_values = np.zeros(len(self.params))

        elif initial_state == 'random':
            initial_values = np.random.uniform(-np.pi, np.pi, len(self.params))

        else:
            raise ValueError('initial_state must be one of equal, zero or random.')

        return initial_values
