from qiskit import QuantumRegister, Aer, assemble
from qiskit.ignis.mitigation import complete_meas_cal, CompleteMeasFitter


def do_measurement_calibration(noise_model, n):
    """
    Creates a measurement calibration fitter as described in
    https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html and the Qiskit Ignis
    tutorial. This should be done once before all circuits since it is potentially expensive.
    :return A measurement calibration fitter. See qiskit's CompleteMeasFitter class.
    """

    print('Calculating measurement calibration')
    qr = QuantumRegister(n)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=None, qr=qr, circlabel='mcal')
    backend = Aer.get_backend('qasm_simulator')
    qobj = assemble(meas_calibs, backend=backend, shots=50)
    job = backend.run(qobj, noise_model=noise_model)
    cal_results = job.result()
    print('Finished calculating measurement calibration')
    return CompleteMeasFitter(cal_results, state_labels, qubit_list=None, circlabel='mcal')


