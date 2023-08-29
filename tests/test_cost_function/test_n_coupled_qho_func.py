from copy import copy
from typing import List
from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.ignis.mitigation import CompleteMeasFitter

from cost_function.n_coupled_qho_func import NCoupledQHOFunc


class TestNCoupledQHOFunc(TestCase):
    def setUp(self) -> None:
        self.default_test_args = {"num_oscillators": 2,
                                  "encoding": "bin",
                                  "ansatz": "vatan-red",
                                  "num_qubits": 4,
                                  "depth": 1,
                                  "gammas": np.array([[0, 1.], [1., 0.]]),
                                  "optimiser": "adam",
                                  "shots": 100,
                                  "backend": "qasm_simulator",
                                  "noise_model": None}

        self.default_cost_func = NCoupledQHOFunc(**self.default_test_args)

    def test_init_backends(self):
        args_no_backend = copy(self.default_test_args)
        args_no_backend.pop("backend")

        NCoupledQHOFunc(**args_no_backend, backend='qasm_simulator')
        NCoupledQHOFunc(**args_no_backend, backend="fake_montreal")
        NCoupledQHOFunc(**args_no_backend, backend="fake_lagos")
        NCoupledQHOFunc(**args_no_backend, backend="statevector_simulator")

        self.assertRaises(NotImplementedError, NCoupledQHOFunc, **args_no_backend, backend="incorrect backend")

    def test_init_noise_models(self):
        args_no_noise_model = copy(self.default_test_args)
        args_no_noise_model.pop("noise_model")

        NCoupledQHOFunc(**args_no_noise_model, noise_model="fake_montreal")
        NCoupledQHOFunc(**args_no_noise_model, noise_model="fake_mumbai")

        self.assertRaises(NotImplementedError, NCoupledQHOFunc, **args_no_noise_model,
                          noise_model="incorrect noise model")
        self.assertRaises(FileNotFoundError, NCoupledQHOFunc, **args_no_noise_model, noise_model="load")

    def test_init_circuits_and_operators(self):
        self.assertIsInstance(self.default_cost_func.transpiled_l, QuantumCircuit)
        self.assertEqual(self.default_cost_func.transpiled_l.num_qubits, 4)

        self.assertIsInstance(self.default_cost_func.h_operator, WeightedPauliOperator)
        self.assertIsInstance(self.default_cost_func.h2_operator, WeightedPauliOperator)
        self.assertIsInstance(self.default_cost_func.h3_operator, WeightedPauliOperator)
        self.assertEqual(self.default_cost_func.h_operator.num_qubits, 4)

    def test_evaluate_cost_function(self):
        params = np.array([[0., 1., 2., 3., 4., 5., 6., 7.]])
        out = self.default_cost_func.evaluate_cost_function(params, self.default_cost_func.h_operator)
        self.assertIsInstance(out, List)
        self.assertIsInstance(out[0], tuple)
        self.assertEqual(len(out[0]), 2)
        self.assertIsInstance(out[0][0], float)
        self.assertIsInstance(out[0][1], float)

    def test_evaluate_cost_function_statevector(self):
        args_no_backend = copy(self.default_test_args)
        args_no_backend.pop("backend")
        cost_func = NCoupledQHOFunc(**args_no_backend, backend="statevector_simulator")
        params = np.array([[0., 1., 2., 3., 4., 5., 6., 7.]])
        out = cost_func.evaluate_cost_function(params, cost_func.h_operator)
        self.assertIsInstance(out, List)
        self.assertIsInstance(out[0], tuple)
        self.assertEqual(len(out[0]), 2)
        self.assertIsInstance(out[0][0], float)
        self.assertIsInstance(out[0][1], float)

        self.assertRaises(AssertionError, lambda: cost_func.evaluate_cost_function(params,
                                                                                   self.default_cost_func.h_operator))

    def test_call_cost_func_object_different_optimisers(self):
        # output of __call__ need to be different for each optimiser
        params = np.array([0., 1., 2., 3., 4., 5., 6., 7.])

        args_no_optimiser = copy(self.default_test_args)
        args_no_optimiser.pop("optimiser")

        out_midaco = NCoupledQHOFunc(**args_no_optimiser, optimiser='midaco')(params)

        self.assertIsInstance(out_midaco, tuple)
        self.assertIsInstance(out_midaco[0], list)
        self.assertIsInstance(out_midaco[1], list)

        out_aqgd = NCoupledQHOFunc(**args_no_optimiser, optimiser='aqgd')(params)
        out_aqgd_fin_diff = NCoupledQHOFunc(**args_no_optimiser, optimiser='aqgd-fin-diff')(params)
        out_cobyla = NCoupledQHOFunc(**args_no_optimiser, optimiser='cobyla')(params)

        for out in [out_aqgd, out_aqgd_fin_diff, out_cobyla]:
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 1)
            self.assertIsInstance(out[0], float)

        out_adam = NCoupledQHOFunc(**args_no_optimiser, optimiser='adam')(params)

        self.assertIsInstance(out_adam, list)
        self.assertEqual(len(out_adam), 1)
        self.assertIsInstance(out_adam[0], tuple)
        self.assertEqual(len(out_adam[0]), 2)

    def test_construct_circuit_operator(self):
        para_dic = {k: l for k, l in zip(self.default_cost_func.all_params,
                                         range(len(self.default_cost_func.all_params)))}
        circuits = self.default_cost_func.construct_circuit_operator(self.default_cost_func.transpiled_l,
                                                                     self.default_cost_func.h_operator,
                                                                     para_dic,
                                                                     self.default_cost_func.backend)

        for circuit in circuits:
            # all parameters are bound and circuit is correct size
            self.assertIsInstance(circuit, QuantumCircuit)
            self.assertEqual(circuit.num_parameters, 0)
            self.assertEqual(circuit.num_qubits, 4)
            self.assertEqual(len(circuit.clbits), 4)

        # one circuit created for each basis measured
        self.assertEqual(len(circuits), len(self.default_cost_func.h_operator.basis))

    def test_create_groups_for_non_interacting_terms(self):
        args_no_n = copy(self.default_test_args)
        args_no_n.pop("num_oscillators")
        args_no_n.pop("num_qubits")
        args_no_n.pop("gammas")

        cost_func = NCoupledQHOFunc(**args_no_n, gammas=np.ones((4, 4)), num_oscillators=4, num_qubits=4)
        grouped_non_int_terms, individual_oscillator_non_int_terms = cost_func.create_groups_for_non_interacting_terms()
        # just one group expected
        self.assertEqual(len(grouped_non_int_terms), 1)
        # four oscillators
        self.assertEqual(len(individual_oscillator_non_int_terms), 4)
        # each oscillator has two terms
        for sublist in individual_oscillator_non_int_terms:
            self.assertEqual(len(sublist), 1)
            self.assertEqual(len(sublist[0]), 2)

    def test_create_groups_for_coupling_terms(self):
        args_no_n = copy(self.default_test_args)
        args_no_n.pop("num_oscillators")
        args_no_n.pop("num_qubits")
        args_no_n.pop("gammas")

        cost_func = NCoupledQHOFunc(**args_no_n, gammas=np.ones((4, 4)), num_oscillators=4, num_qubits=8)
        one_factorisation_groups, individual_interaction_groups = cost_func.create_groups_for_coupling_terms()

        # six interactions
        self.assertEqual(len(individual_interaction_groups), 6)

        for sublist in individual_interaction_groups:
            # 16 total terms per interaction
            subtot = sum(len(subsublist) for subsublist in sublist)
            self.assertEqual(subtot, 16)

        # total number of terms should be same after grouping
        self.assertEqual(sum(len(sublist) for sublist in one_factorisation_groups),
                         6 * 16)

    def test_perform_repeated_measurements(self):
        params = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        exps, stdevs = self.default_cost_func.perform_repeated_measurements(params, 4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(len(stdevs), 4)

    def test_lanczos_mitigation(self):
        params = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        self.assertRaises(NotImplementedError,
                          lambda: self.default_cost_func.lanczos_mitigation(params, num_measurements=2,
                                                                            save_output=False))

    def test_subtract_uncoupled_noise(self):
        params = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        params_zero = np.zeros(8)
        subtract_out = self.default_cost_func.subtract_uncoupled_noise(params, params_zero, 2)
        self.assertEqual(len(subtract_out), 2)
        self.assertIsInstance(subtract_out[0], float)  # exp
        self.assertIsInstance(subtract_out[1], float)  # stdev

    def test_evaluate_state(self):
        params = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        state = self.default_cost_func.evaluate_state(params)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2 ** 4,))
        self.assertEqual(state.T.conj() @ state, 1.)

    def test_do_measurement_calibration(self):
        meas_cal = self.default_cost_func.do_measurement_calibration()
        self.assertIsInstance(meas_cal, CompleteMeasFitter)
