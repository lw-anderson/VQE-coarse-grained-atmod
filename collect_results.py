import argparse
import json
import os

import mpltex
import numpy as np

from read_outputs import read_outputs


def collect_results(results_directory, all_plots=True):
    """
    Collect all results from a given directory. Will run read_outputs to get gammas, minimum, variance and overlap.
    """
    print('Collecting results from directory:')
    print(results_directory)
    results = {}
    for output in sorted(os.listdir(results_directory)):
        if os.path.isdir(os.path.join(results_directory, output)):
            (minimum_mean, mean_expected_repeated_evals_stdev, mean_repeated_evals_stdev), exact_final_output, \
                (lanczos_mean, lanczos_stdev), (extrap_then_lanczos_mean, extrap_then_lanczos_stdev), \
                (lanczos_then_extrap_mean, lanczos_then_extrap_stdev), (
            subtracted_mean, subtracted_stdev), overlap, args \
                = read_outputs(os.path.join(results_directory, output), all_plots=all_plots)
            key = 'VQA depth ' + str(args.depth)

            varied_param = args.ext_field if args.varied_param == "ext-field" else args.gammas

            if type(varied_param) == str:
                varied_param = eval(varied_param)

            if args.geometry == "regular_polygon" and not args.varied_param == "ext-field":
                # interior_angle = (args.num_oscillators - 2) * np.pi / args.num_oscillators
                alpha_over_diameter_cubed = np.array(varied_param) * np.sin(np.pi / args.num_oscillators) ** 3
                varied_param = alpha_over_diameter_cubed

            if key in results:
                results[key].append((varied_param, minimum_mean, mean_repeated_evals_stdev,
                                     lanczos_mean, lanczos_stdev,
                                     extrap_then_lanczos_mean, extrap_then_lanczos_stdev,
                                     lanczos_then_extrap_mean, lanczos_then_extrap_stdev,
                                     subtracted_mean, subtracted_stdev,
                                     overlap, exact_final_output))
            else:
                results[key] = [(varied_param, minimum_mean, mean_repeated_evals_stdev,
                                 lanczos_mean, lanczos_stdev,
                                 extrap_then_lanczos_mean, extrap_then_lanczos_stdev,
                                 lanczos_then_extrap_mean, lanczos_then_extrap_stdev,
                                 subtracted_mean, subtracted_stdev,
                                 overlap, exact_final_output)]
    return results


def collect_reduced_results(results_directory):
    """
    Collect only cost function results from a given directory. Does not run read_outputs and only needs access to
    costfunc_values.txt and arguments.json.
    """
    print('Collecting results from directory:')
    print(results_directory)
    results = {}
    for output in sorted(os.listdir(results_directory)):
        if os.path.isdir(os.path.join(results_directory, output)):
            args = argparse.Namespace(**json.load(open(os.path.join(results_directory, output, 'arguments.json'), "r")))
            costfunc_values = np.loadtxt(os.path.join(results_directory, output, 'costfunc_values.txt'))
            minimum = costfunc_values[-1]
            variance = 0
            overlap = 0
            key = str(args.ansatz) + ' depth ' + str(args.depth)
            if key in results:
                results[key].append((eval(args.gammas), minimum, variance, overlap))
            else:
                results[key] = [(eval(args.gammas), minimum, variance, overlap)]
    return results


def add_results_to_plot(results, energy_ax, varied_gamma_pos=0, correction=0, error_ax=None, real_val_func=None):
    linestyles = mpltex.linestyle_generator(lines=[], hollow_styles=[])
    for label in results:
        linestyle1 = next(linestyles)
        linestyle2 = {}
        linestyle3 = {}
        linestyle4 = {}
        linestyle5 = {}

        energy_ax.plot([], [], label=label, **linestyle1)

        label_exact = False
        label_lanczos = False
        for (gammas, minimum_mean, mean_repeated_evals_variance, lanczos_mean, lanczos_stdev, extrap_then_lanczos_mean,
             extrap_then_lanczos_stdev, lanczos_then_extrap_mean, lanczos_then_extrap_stdev, subtracted_mean,
             subtracted_stdev, overlap, exact_final_output) in results[label]:
            energy_ax.errorbar([gammas[varied_gamma_pos]], [minimum_mean - correction],
                               [mean_repeated_evals_variance ** 0.5],
                               **linestyle1, markersize=3)
            if exact_final_output is not None:
                linestyle2 = next(linestyles) if not linestyle2 else {}
                energy_ax.plot([gammas[varied_gamma_pos]], [exact_final_output[0]], **linestyle2)
                label_exact = True
            if lanczos_mean is not None and lanczos_stdev is not None:
                linestyle3 = next(linestyles) if not linestyle3 else {}
                energy_ax.errorbar([gammas[varied_gamma_pos]], [lanczos_mean], [lanczos_stdev],
                                   **linestyle3, markersize=3)
                label_lanczos = True
            if extrap_then_lanczos_mean is not None and extrap_then_lanczos_stdev is not None:
                linestyle4 = next(linestyles) if not linestyle4 else {}
                energy_ax.errorbar([gammas[varied_gamma_pos]], [extrap_then_lanczos_mean], [extrap_then_lanczos_stdev],
                                   **linestyle4, markersize=3)
            if lanczos_then_extrap_mean is not None and lanczos_then_extrap_stdev is not None:
                linestyle5 = next(linestyles) if not linestyle5 else {}
                energy_ax.errorbar([gammas[varied_gamma_pos]], [lanczos_then_extrap_mean], [lanczos_then_extrap_stdev],
                                   **linestyle5, markersize=3)

            if error_ax and real_val_func is not None:
                real_val = real_val_func(gammas)
                error_ax.errorbar([gammas[varied_gamma_pos]], [minimum_mean - real_val],
                                  [mean_repeated_evals_variance ** 0.5],
                                  **linestyle1, markersize=3)

        if label_exact:
            energy_ax.plot([], [], label='VQA (exact output)', **linestyle2, markersize=3)
        if label_lanczos:
            energy_ax.plot([], [], label='Lanczos', **linestyle3, markersize=3)
            energy_ax.plot([], [], label='Extrap + Lanczos', **linestyle4, markersize=3)
            energy_ax.plot([], [], label='Lanczos + Exact', **linestyle5, markersize=3)


def add_results_to_plot_chosen_x_axis(results, energy_ax, infidelity_ax, varied_gamma_pos,
                                      gamma_values, x_values):
    linestyles = mpltex.linestyle_generator(lines=[], hollow_styles=[])
    for label in results:
        linestyle = next(linestyles)
        energy_ax.plot([], [], label=label, **linestyle)
        if infidelity_ax is not None:
            infidelity_ax.plot([], [], label=label, **linestyle)
        for (gammas, minimum, variance, overlap) in results[label]:
            i = gamma_values.index(gammas[varied_gamma_pos])
            x_val = x_values[i]
            energy_ax.errorbar([x_val], [minimum], [variance ** 0.5], **linestyle)
            if infidelity_ax is not None:
                infidelity_ax.plot([x_val], [1 - overlap], **linestyle)


def add_results_to_plot_three_body_only(results, energy_ax, infidelity_ax, varied_gamma_pos,
                                        gamma_values, x_values):
    linestyles = mpltex.linestyle_generator(lines=[], hollow_styles=[])
    for label in results:
        linestyle = next(linestyles)
        # energy_ax.plot([], [], label='VQA', **linestyle)
        if infidelity_ax is not None:
            infidelity_ax.plot([], [], label=label, **linestyle)
        for (gammas, minimum, variance, overlap) in results[label]:

            two_body_energy = sum([(1 + gam / 2) ** 0.5 + (1 - gam / 2) ** 0.5 for gam in gammas]) \
                              - len(gammas)

            i = gamma_values.index(gammas[varied_gamma_pos])
            x_val = x_values[i]
            energy_ax.errorbar([x_val], [minimum - two_body_energy], [variance ** 0.5], **linestyle)
            if infidelity_ax is not None:
                infidelity_ax.plot([x_val], [1 - overlap], **linestyle)
