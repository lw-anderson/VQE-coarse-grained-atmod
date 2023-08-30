import argparse
import json
import logging
import os
from copy import copy

import numpy as np
from numpy.linalg import eig

from cost_function.cost_func_factory import CostFuncFactory
from utils import entanglement_entropy, produce_all_plots

# 867 is number of circuits for 17 lots of 6 VQE points (inc. gamma=0)
max_evals_at_once = 867


def load_previous_output(output_directory):
    owd = os.getcwd()
    os.chdir(output_directory)
    if os.path.isfile("costfunc_values.txt") and os.path.isfile("parameters_values.txt"):
        cost_function_values = np.loadtxt("costfunc_values.txt")
        x_values = np.loadtxt("parameters_values.txt")

    elif os.path.isfile("MIDACO_HISTORY.TXT"):
        midaco_output = np.loadtxt("MIDACO_HISTORY.TXT", skiprows=12)
        cost_function_values = midaco_output[:, 0]
        x_values = midaco_output[:, 1:]

    elif any(fname.startswith('temp-evals ') for fname in os.listdir('./')) and \
            any(fname.startswith('temp-params ') for fname in os.listdir('./')):
        cost_fname, x_fname = None, None
        for fname in os.listdir('./'):
            if fname.startswith('temp-evals'):
                cost_fname = fname
            elif fname.startswith('temp-params'):
                x_fname = fname
        print(f'Loading incomplete optimisation outputs from {cost_fname} and {x_fname}')
        cost_function_values = np.array(np.load(cost_fname))
        x_values = np.load(x_fname)
    else:
        raise FileNotFoundError("Could not find output file.")

    runtime_args = json.load(open("arguments.json", "r"))
    runtime_args = argparse.Namespace(**runtime_args)

    os.chdir(owd)

    return cost_function_values, x_values, runtime_args


def read_outputs(output_directory, all_plots=True):
    print(output_directory)
    owd = os.getcwd()
    logging.info('READING OUTPUTS AND PRODUCING PLOTS')
    args = argparse.Namespace(**json.load(open(os.path.join(output_directory, 'arguments.json'), "r")))
    args.meas_mit = False
    max_evals = [args.maxeval]
    for args_file in sorted([f for f in os.listdir(output_directory) if f[:10] == 'arguments_'],
                            key=lambda x: int(x[10:-5])):
        args = argparse.Namespace(**json.load(open(os.path.join(output_directory, args_file), "r")))
        max_evals.append(args.maxeval)
    cumulative_evals = [sum(max_evals[:i + 1]) for i, _ in enumerate(max_evals)]

    cost_function_values, x_values, runtime_args = load_previous_output(output_directory)

    os.chdir(output_directory)

    problem_func = CostFuncFactory(runtime_args).get()

    if type(runtime_args.gammas) == str:
        runtime_args.gammas = eval(runtime_args.gammas)

    if args.solver in ['midaco']:
        (minimum_optimised, position) = min((mean, idx) for (idx, (mean, stdev)) in enumerate(cost_function_values))
        x_optimised = x_values[position, :]
        x_values = None
    elif args.solver in ['cobyla']:
        position = -1
        minimum_optimised = cost_function_values[position][0]
        x_optimised = x_values[position, :]
        x_values = None
    else:
        position = -1
        minimum_optimised = cost_function_values[position][0]
        x_optimised = x_values[position, :]

    eigenvalues, eigenvectors = eig(problem_func.hamiltonian)
    minimum_reference, state_reference = min((val, vec) for (val, vec) in zip(eigenvalues, eigenvectors.T))

    minimum_analytic = problem_func.analytic_minimum

    state_optimised = problem_func.evaluate_state(x_optimised)
    state_optimised = np.sign(state_optimised[0]) * state_optimised

    print('Entanglement entropies for states')
    if args.num_oscillators > 1:
        for oscillator in range(args.num_oscillators):
            ent_optimised_state = entanglement_entropy(state_optimised, args.num_qubits // args.num_oscillators,
                                                       args.num_oscillators, oscillator)
            ent_reference_state = entanglement_entropy(state_reference, args.num_qubits // args.num_oscillators,
                                                       args.num_oscillators, oscillator)

            print(f'       qho {oscillator} | ref state: {round(ent_reference_state, 5)} | '
                  f'opt state: {round(ent_optimised_state, 5)}')

    if all_plots:
        print("making plots")

        if args.backend != 'statevector_simulator':
            args_with_sv = copy(args)
            args_with_sv.backend = 'statevector_simulator'
            cost_func_sv = CostFuncFactory(args_with_sv, save_temp_evals=False).get()
            exact_cost_function_values = []
            for x in x_values:
                exact_cost_function_values.append(cost_func_sv(x)[0])
        else:
            exact_cost_function_values = None

        produce_all_plots(runtime_args, state_optimised, minimum_optimised, state_reference, minimum_reference,
                          cost_function_values, exact_cost_function_values, x_values, minimum_analytic,
                          (minimum_reference - 0.05, minimum_reference + 0.2),
                          save=True, eval_routine=cumulative_evals)
        print("made plots")

    os.chdir(owd)
    return


def list_tuples_to_array(tuple_list):
    return np.array([list(tuple) for tuple in tuple_list])


def measure_all_outputs_concurrently(root_dir, num_repeats):
    args_params_ops = []
    for sub_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, sub_dir)):
            args_fname = os.path.join(root_dir, sub_dir, 'arguments.json')
            args = argparse.Namespace(**json.load(open(args_fname)))
            param_fname = os.path.join(root_dir, sub_dir, 'parameters_values.txt')
            x = np.loadtxt(param_fname)[-1]

            temp_args = copy(args)
            temp_args.meas_cal = False  # Avoiding unnecessary measurement calibration
            problem_func = CostFuncFactory(temp_args).get()

            ops = (problem_func.h_operator,)
            args_params_ops.append([sub_dir, args, x, ops])

    args_params_ops.sort(key=lambda lst: abs(lst[1].gammas[0]))

    for [sub_dir, args, _, _] in args_params_ops[1:]:
        temp_args = copy(args)
        temp_args.gammas = args_params_ops[0][1].gammas
        if temp_args != args_params_ops[0][1]:
            logging.warning(f'args for sub dir {sub_dir} differ from those of {args_params_ops[0][0]}.')

    problem_func = CostFuncFactory(args_params_ops[0][1]).get()

    combined_parameters = []
    operators = []
    for i in range(num_repeats):
        for [_, _, x, ops] in args_params_ops:
            for op in ops:
                combined_parameters.append(x)
                operators.append(op)

    outputs, overlaps \
        = problem_func.evaluate_cost_function(combined_parameters, operators, return_zero_state_overlap=True,
                                              max_evals_at_once=max_evals_at_once)

    results = {}
    for i, [sub_dir, args, _, _] in enumerate(args_params_ops):
        gamma = args.gammas[0]
        results[sub_dir] = (gamma,
                            outputs[i::(len(args_params_ops))],  # h1results
                            overlaps[i::(len(args_params_ops))])

    np.save(os.path.join(root_dir, 'subtraction_outputs'), results, allow_pickle=True)
    np.save(os.path.join(root_dir, 'args_params_ops'), args_params_ops, allow_pickle=True)
    return results


def main(args):
    for file in sorted(os.listdir(args.directory)):
        if os.path.isdir(os.path.join(args.directory, file)):
            read_outputs(os.path.join(args.directory, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load outputs of optimisation routine and produce plots.')
    parser.add_argument('--directory',
                        default='./path/to/output/',
                        type=str, help='Directory of output to read.')

    read_args = parser.parse_args()
    main(read_args)
