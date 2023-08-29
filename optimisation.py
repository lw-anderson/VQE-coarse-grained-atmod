import json
import logging
import os

import numpy as np
from qiskit_nature.algorithms.pes_samplers import Extrapolator

from calculate_gammas import get_ring_gammas, get_polygon_gammas
from cost_function.cost_func_factory import CostFuncFactory
from optimisation_parser import OptimisationParser
from read_outputs import load_previous_output, read_outputs
from solver.solver_factory import SolverFactory
from utils import create_output_path, create_json_filename


def main(args):
    solver = SolverFactory(args).get()

    run_directory = os.getcwd()
    output_directories = []
    opt_params = dict()
    args.gammas = eval(args.gammas)
    if args.varied_param == "ext-field":
        if not (type(args.gammas) is list and (len(args.gammas) == 0 or type(args.gammas[0]) is float)):
            raise ValueError("gammas must be list of floats (or empty if one oscillator).")
        if not (type(args.ext_field) is list and type(args.ext_field[0]) is float):
            raise ValueError("ext_field must be list of floats.")
        hamiltonian_param_values_list = args.ext_field
        args.gammas = args.gammas

    elif args.varied_param == "gammas":
        if args.geometry:
            if not (type(args.gammas) is list and type(args.gammas[0]) is float):
                raise ValueError("if using fixed geometry, gammas must be list of floats.")
        else:
            if not (type(args.gammas) is list and type(args.gammas[0]) is list):
                raise ValueError("if not using fixed geometry, gammas must be list of lists.")

        if not type(args.ext_field) is float:
            raise ValueError("ext_field must float.")
        hamiltonian_param_values_list = args.gammas
        args.ext_field = args.ext_field
    else:
        raise ValueError("varied_param is not allowed, must be one of 'gammas' or 'ext_field'")

    if args.restart:
        output_directory_prefix = args.restart
        for sub_dir in os.listdir(args.restart):
            output_directory = os.path.join(args.restart, sub_dir)
            cost_function_values, x_values, runtime_args = load_previous_output(output_directory)
            output_directories.append(output_directory)
            opt_params[runtime_args.gammas[0]] = x_values[-1]
    else:
        output_directory_prefix = create_output_path(args)
    read_outputs_at_end = args.backend in ['statevector_simulator', 'qasm_simulator']

    extrapolator = Extrapolator.factory('poly')

    for hamiltonian_param_value in hamiltonian_param_values_list:
        if args.varied_param == "gammas":
            gammas = hamiltonian_param_value
            # overwrite gammas dependent on specified oscillator geometry
            if args.geometry == 'ring':
                if type(hamiltonian_param_value) is not float:
                    raise ValueError('if using certain geometry, gammas should be float.')
                gamma_0 = gammas
                gammas = get_ring_gammas(gamma_0, args.num_oscillators)
                gammas = list(gammas[np.triu_indices(gammas.shape[0], k=1)])

            elif args.geometry == 'regular_polygon':
                if type(gammas) is not float:
                    raise ValueError('if using certain geometry, gammas should be float.')
                gamma_0 = gammas
                gammas = get_polygon_gammas(gamma_0, args.num_oscillators)
                gammas = list(gammas[np.triu_indices(gammas.shape[0], k=1)])
            else:
                if type(gammas) is not list:
                    raise ValueError('if not using certain geometry, gammas should be list of floats.')

            rounded_gammas = [round(gamma, 3) for gamma in gammas]
            result_subdirectory = os.path.join(os.getcwd(), output_directory_prefix, f'{rounded_gammas}')
            output_directories.append(result_subdirectory)
            os.makedirs(result_subdirectory)
            os.chdir(result_subdirectory)
            args.gammas = gammas

        elif args.varied_param == "ext-field":
            field = hamiltonian_param_value
            rounded_field = round(field, 3)
            result_subdirectory = os.path.join(output_directory_prefix, f'{rounded_field}')
            output_directories.append(result_subdirectory)
            os.makedirs(result_subdirectory)
            os.chdir(result_subdirectory)
            args.ext_field = field
        else:
            raise ValueError("varied_param_type must be 'gamma' or 'ext-field'.")

        # Create an object of the cost function class for an Ansatz with n qubits and depth d
        problem_func = CostFuncFactory(args).get()

        prev_points = list(opt_params.keys())
        prev_params = list(opt_params.values())
        n_pp = len(prev_points)

        if prev_points:
            if n_pp <= 2:
                distances = np.array(hamiltonian_param_value) - \
                            np.array(prev_points).reshape(n_pp, -1)
                # find min 'distance' from point to previous points
                min_index = np.argmin(np.linalg.norm(distances, axis=1))
                # update initial point
                initial_point = prev_params[min_index]  # type: ignore
            else:  # extrapolate using saved parameters
                points = [gammas[0] for gammas in hamiltonian_param_values_list] if type(
                    hamiltonian_param_values_list[0]) is list else hamiltonian_param_values_list
                param_sets = extrapolator.extrapolate(points=points, param_dict=opt_params)
                # update initial point, note param_set is a dictionary
                initial_point = param_sets.get(hamiltonian_param_value[0] if type(
                    hamiltonian_param_values_list[0]) is list else hamiltonian_param_value)  # type: ignore

        else:
            initial_point = problem_func.ansatz.get_initial(initial_state='random')

        logging.info('Performing optimisation')
        logging.info('Saving outputs to ' + output_directory_prefix)
        arguments_fname = create_json_filename()
        with open(arguments_fname, "w") as fp:
            json.dump(vars(args), fp)

        # Call the optimisation algorithm
        x_optimised, minimum_optimised = solver(problem_func, initial_point)
        opt_params[hamiltonian_param_value[0] if type(
            hamiltonian_param_value) is list else hamiltonian_param_value] = x_optimised

        if args.read_outputs and not read_outputs_at_end:
            read_outputs(result_subdirectory, all_plots=False)

        os.chdir(run_directory)

    if args.read_outputs and read_outputs_at_end:
        for result_subdirectory in output_directories:
            read_outputs(result_subdirectory)
    print(output_directories)
    return 0


if __name__ == '__main__':
    parser = OptimisationParser()

    opt_args = parser.parse_args()

    main(opt_args)
