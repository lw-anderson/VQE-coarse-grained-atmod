from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import RemoveFinalMeasurements


def repeat_cnots(qc: QuantumCircuit, num_cnots: int) -> QuantumCircuit:
    """
    Return circuit with each CNOT replaced by 'num_cnots' number of CNOTS. num_cnots must be one of 1,3 or 5 such that
    new circuit implements same operation as original circuit assuming no noise.
    """
    if num_cnots not in [1, 3, 5]:
        raise ValueError('num_cnots must be one of 1, 3, 5')

    if num_cnots == 1:
        return qc

    if len(qc.cregs) > 0:
        qc_new = QuantumCircuit(qc.qregs[0], qc.cregs[0], name=qc.name)
    else:
        qc_new = QuantumCircuit(qc.qregs[0])
    for gate in qc:
        if type(gate[0]) is CXGate:
            for _ in range(num_cnots):
                qc_new.append(gate[0], gate[1], gate[2])
                qc_new.barrier(gate[1])
        else:
            qc_new.append(gate[0], gate[1], gate[2])
    return qc_new


def fold_circuit(qc: QuantumCircuit, num_folds: int) -> QuantumCircuit:
    """
    Return circuit repeated one after another 'num_folds' times (alternating original and adjoint of circuit).
    num_folds must be one of 1,3 or 5 such that new circuit implements same operation as original circuit assuming no
    noise.
    """
    if num_folds not in [1, 3, 5]:
        raise ValueError('num_folds must be one of 1, 3, 5')

    if num_folds == 1:
        return qc

    if len(qc.cregs) > 0:
        qc_new = QuantumCircuit(qc.qregs[0], qc.cregs[0], name=qc.name)
        qc_no_meas = dag_to_circuit(RemoveFinalMeasurements().run(circuit_to_dag(qc)))
        qc_no_meas.add_register(qc.cregs[0])
        qc_inv = qc_no_meas.inverse()
        for i in range(num_folds - 1):
            if i % 2 == 0:
                qc_new.append(qc_no_meas.to_instruction(), qc_no_meas.qregs[0], qc_no_meas.cregs[0])
            else:
                qc_new.append(qc_inv.to_instruction(), qc_inv.qregs[0], qc_inv.cregs[0])
            qc_new.barrier(qc_new.qregs[0])
        qc_new.append(qc, qc.qregs[0], qc.cregs[0])
    else:
        qc_new = QuantumCircuit(qc.qregs[0], name=qc.name)
        qc_no_meas = dag_to_circuit(RemoveFinalMeasurements().run(circuit_to_dag(qc)))
        qc_inv = qc_no_meas.inverse()
        for i in range(num_folds - 1):
            if i % 2 == 0:
                qc_new.append(qc_no_meas.to_instruction(), qc_no_meas.qregs[0])
            else:
                qc_new.append(qc_inv.to_instruction(), qc_inv.qregs[0])
            qc_new.barrier(qc_new.qregs[0])
        qc_new.append(qc, qc.qregs[0])
    return qc_new.decompose()
