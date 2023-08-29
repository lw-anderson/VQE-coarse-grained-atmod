import argparse
import json
import os

import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from numpy.linalg import eig

from collect_results import collect_results
from operators.hamiltonian import get_extended_hamiltonian
from utils import set_plot_style

set_plot_style(9)
figsize = (3.375, 2.25)

linestyles = mpltex.linestyle_generator(lines=[], markers=['o', 's', 'v'], hollow_styles=[False])
linestyles_sub = mpltex.linestyle_generator(lines=[], markers=['o', 's', 'v'], hollow_styles=[False])

angstrom = 1e-10

# Ne Dimer
cov_radius = 38e-12 / angstrom
VdW_radius = 154e-12 / angstrom
alpha = 0.204

# I2 Dimer
cov_radius = 1.33e-12 / angstrom  # In angstroms (A)
VdW_radius = 198e-12 / angstrom  # A
alpha = 14.5  # A^3
hbar_omega = 9.6106  # eV

# # Cs2 Dimer
# VdW_radius = 3.7/2  # A
# alpha = 8.8  # A^3
#
# # PBr3 Dimer
# PP_dist = 5 # (A) For P-P distance
# BrBr_dist = 3.5 # (A) For Br-Br distance
# alpha = 14.5

# num oscillators and qubits per oscillator
n = 2
m = 2


@np.vectorize
def gamma_to_dist(gamma):
    """
    Returns R in units of Bohr radius for a given gamma.
    """
    if gamma == 0.0:
        return 10
    return (4 * alpha / abs(gamma)) ** (1 / 3)


@np.vectorize
def dist_to_gamma(dist):
    """
    Returns gamma for a given dist (in units of Bohr radius)
    """
    return 4 * alpha / (dist ** 3)


@np.vectorize
def convert_energy_dimensions(energy):
    """
    Returns energy in units of eV when input in units of ℏω/2
    """
    return hbar_omega * energy


def exact_energy(gamma, n, m):
    return np.sqrt(1 + gamma / 2) * (2 * n + 1) + np.sqrt(1 - gamma / 2) * (2 * m + 1)


def calc_exact_energies(n, m, gammas):
    return [exact_energy(gamma, n, m) for gamma in gammas]


def add_exact_energies_to_plot(n_plus_m_max, gammas, axes):
    kwargs = {'color': 'r'}
    for n_plus_m in range(n_plus_m_max):
        for n in range(n_plus_m + 1):
            m = n_plus_m - n
            exact_energies = calc_exact_energies(n, m, gammas)
            for ax in axes:
                ax.plot(gammas, exact_energies, **kwargs)
    # plt.plot([], [], label='All Fock states', **kwargs)
    return None


def truncated_energies_pair(gamma, n, anharmonic_kwargs):
    hamiltonian = np.float64(get_extended_hamiltonian(2, n, np.array([[0, gamma], [gamma, 0]]), encoding='bin',
                                                      numpy_array=True, **anharmonic_kwargs))
    eval, evec = eig(hamiltonian)
    ev_list = [tup for tup in zip(eval, evec)]
    ev_list.sort(key=lambda tup: tup[0], reverse=False)
    eval, evec = zip(*ev_list)
    return eval


def calc_truncated_energies_pair(gammas, n, anharmonic_kwargs):
    eigenvalues = np.array([truncated_energies_pair(gamma, n, anharmonic_kwargs) for gamma in gammas])
    return eigenvalues.transpose()


def add_truncated_energies_to_plot(gammas, axes, anharmonic_kwargs: dict = {}):
    kwargs = {'color': 'k', 'linestyle': '--'}
    for energies in calc_truncated_energies_pair(gammas, 4, anharmonic_kwargs):
        for ax in axes:
            ax.plot(gammas, energies, **kwargs)
    plt.plot([], [], label=r'$d=4$ Truncated Fock states', **kwargs)  # Add single legend item
    return None


def add_exact_energies_to_distance_plot(n_plus_m_max, distances, axes, convert_energy=True):
    kwargs = {'color': 'k', 'linewidth': 1}
    gammas = [dist_to_gamma(dist) for dist in distances]
    for n_plus_m in range(n_plus_m_max):
        for n in range(n_plus_m + 1):
            m = n_plus_m - n
            exact_energies = calc_exact_energies(n, m, gammas)
            for ax in axes:
                ax.plot(distances, convert_energy_dimensions(np.array(exact_energies) - 2), **kwargs, zorder=-100,
                        label='Analytic')
    # plt.plot([], [], label='All Fock states', **kwargs)
    return None


def add_truncated_energies_to_distance_plot(distances, axes, anharmonic_kwargs: dict = {}):
    kwargs = {'color': 'k'}
    gammas = [dist_to_gamma(dist) for dist in distances]
    uncoupled_energy = calc_truncated_energies_pair([0], 4, anharmonic_kwargs)[0]
    for energies in calc_truncated_energies_pair(gammas, 4, anharmonic_kwargs):
        for ax in axes:
            ax.plot(distances, convert_energy_dimensions(np.array(energies) - uncoupled_energy), **kwargs)
    plt.plot([], [], label=r'$d=4$ Truncated Fock states', **kwargs)  # Add single legend item
    return None


def add_results_to_dist_plot(results, energy_ax, varied_gamma_pos=0):
    # which points to plot
    next(linestyles)
    linestyle_exp = False  # next(linestyles)
    linestyle_exact = next(linestyles)
    linestyle_lanczos_extrap = False  # next(linestyles)
    linestyle_lanczos = False  # next(linestyles)
    linestyle_extrap_lanczos = False  # next(linestyles)
    linestyle_subtracted = False  # next(linestyles)

    for label in results:
        for (gammas, minimum_mean, mean_repeated_evals_stdev, lanczos_mean, lanczos_stdev,
             extrap_then_lanczos_mean, extrap_then_lanczos_stdev,
             lanczos_then_extrap_mean, lanczos_then_extrap_stdev,
             subtracted_mean, subtracted_stdev,
             overlap, exact_final_output) in results[label]:

            if linestyle_exp:
                energy_ax.errorbar([gamma_to_dist(gammas[varied_gamma_pos])],
                                   convert_energy_dimensions([minimum_mean - 2]),
                                   convert_energy_dimensions([mean_repeated_evals_stdev]),
                                   **linestyle_exp, capsize=3)
            if linestyle_exact:
                energy_ax.plot([gamma_to_dist(gammas[varied_gamma_pos])],
                               convert_energy_dimensions([exact_final_output[0] - 2]),
                               **linestyle_exact, zorder=-32)
            if linestyle_extrap_lanczos:
                energy_ax.errorbar([gamma_to_dist(gammas[varied_gamma_pos])],
                                   convert_energy_dimensions([extrap_then_lanczos_mean - 2]),
                                   [extrap_then_lanczos_stdev], **linestyle_extrap_lanczos, capsize=3)
            if linestyle_lanczos:
                energy_ax.errorbar([gamma_to_dist(gammas[varied_gamma_pos])],
                                   convert_energy_dimensions([lanczos_mean - 2]),
                                   convert_energy_dimensions([lanczos_stdev]),
                                   **linestyle_lanczos, capsize=3)
            if linestyle_lanczos_extrap:
                energy_ax.errorbar([gamma_to_dist(gammas[varied_gamma_pos])],
                                   convert_energy_dimensions([lanczos_then_extrap_mean - 2]),
                                   convert_energy_dimensions([lanczos_then_extrap_stdev]),
                                   **linestyle_lanczos_extrap, capsize=3)
            if linestyle_subtracted:
                energy_ax.errorbar([gamma_to_dist(gammas[varied_gamma_pos])],
                                   convert_energy_dimensions([subtracted_mean]),
                                   convert_energy_dimensions([subtracted_stdev]),
                                   **linestyle_subtracted, capsize=3)

        if linestyle_exp:
            energy_ax.plot([], [], **linestyle_exp, label='Experiment')
        if linestyle_extrap_lanczos:
            energy_ax.plot([], [], **linestyle_extrap_lanczos, label='Extrap+Lanczos')
        if linestyle_lanczos:
            energy_ax.plot([], [], **linestyle_lanczos, label='Lanczos')
        if linestyle_lanczos_extrap:
            energy_ax.plot([], [], **linestyle_lanczos_extrap, label='Lanczos+Extrap')
        if linestyle_exact:
            energy_ax.plot([], [], **linestyle_exact, label='Noise-free evaluation')
        if linestyle_subtracted:
            energy_ax.plot([], [], **linestyle_subtracted, label='Subtracted')


def add_subtraction_results_to_dist_plots(results, energy_ax):
    linestyle_exp = False  # next(linestyles)
    linestyle_sub = next(linestyles_sub)
    dists, means, stdevs = [], [], []
    E0_mean, E0_stdev = None, None
    zero_state_overlap_mean, zero_state_overlap_stdev = None, None

    for key in results:
        vals = results[key]
        dists.append(gamma_to_dist(vals[0]))
        measurements = np.array(vals[1])
        means.append(np.mean(measurements[:, 0]) - 2)
        stdevs.append(np.std(measurements[:, 0]) / np.sqrt(len(measurements[:, 1])))
        if vals[0] == 0.0:
            key_for_zero = key
            E0_mean = np.mean(measurements[:, 0]) - 2
            E0_stdev = np.std(measurements[:, 0]) / np.sqrt(len(measurements[:, 1]))
            zero_state_overlap_mean = np.mean(vals[-1])
            zero_state_overlap_stdev = np.std(vals[-1]) / np.sqrt(len(measurements[:, 1]))

    if not zero_state_overlap_mean:
        raise FileNotFoundError("Could not find gamma = 0.0 output required for subtraction and scaling.")

    ####### DOING EACH RUN SEPARATELY AND THEN AVERAGE ##########

    # Er_minus_E0_normalised, Er_minus_E0_normalised_stdev = [], []
    # for key in results:
    #     Er_minus_E0_normalised_single_R = []
    #     for Er, E0, zero_overlap in zip(results[key][1], results[key_for_zero][1], results[key_for_zero][-1]):
    #         lam = (1 - zero_overlap) / (1 - 2 ** (-m * n))
    #         Er_minus_E0_normalised_single_R.append((Er[0] - E0[0]) / (1 - lam))
    #     Er_minus_E0_normalised.append(np.mean(Er_minus_E0_normalised_single_R))
    #     Er_minus_E0_normalised_stdev.append(
    #         np.std(Er_minus_E0_normalised_single_R) / np.sqrt(len(Er_minus_E0_normalised_single_R)))

    ####### USING AVERAGES THROUGHOUT INSTEAD ##########

    Er_minus_E0 = np.array(means) - E0_mean
    Er_minus_E0_stdev = np.sqrt(np.array(stdevs) ** 2 + E0_stdev ** 2)

    lam = (1 - zero_state_overlap_mean) / (1 - 2 ** (-m * n))
    lam_stdev = abs(1 / (1 - 2 ** (-m * n))) * zero_state_overlap_stdev

    Er_minus_E0_normalised = Er_minus_E0 / (1 - lam)

    Er_minus_E0_normalised_stdev = np.sqrt((Er_minus_E0_normalised * lam_stdev / (1 - lam)) ** 2
                                           + (Er_minus_E0_stdev / (1 - lam)) ** 2)

    ###########################

    if linestyle_exp:
        energy_ax.errorbar(dists, convert_energy_dimensions(means), convert_energy_dimensions(stdevs),
                           **linestyle_exp, capsize=3)
        energy_ax.plot([], [], **linestyle_exp, label='Raw')
    if linestyle_sub:
        energy_ax.errorbar(dists, convert_energy_dimensions(Er_minus_E0_normalised),
                           convert_energy_dimensions(Er_minus_E0_normalised_stdev),
                           **linestyle_sub, capsize=3, markersize=4, label=r'\textit{ibm\_lagos} w/ subtraction')


def main():
    num_points = 100

    results_dir = '/home/lewis/git/atomisticmodelling/results/2_oscillators_final_results'
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    results = collect_results(results_dir)

    xmin = 3.0
    xmax = 6.0
    ymin = -2.25
    ymax = 0.25

    distances = np.linspace(xmin - 1, xmax + 1, num_points)

    subtraction_results = \
        np.load(os.path.join(results_dir, 'subtraction_outputs.npy'), allow_pickle=True)[()]

    add_exact_energies_to_distance_plot(1, distances, [ax])

    args_file = os.path.join(results_dir, sorted(os.listdir(results_dir))[0], "arguments.json")
    args = argparse.Namespace(**json.load(open(args_file, "r")))

    add_truncated_energies_to_distance_plot(distances, [ax],
                                            {"cubic_prefactor": args.cubic,
                                             "quartic_prefactor": args.quartic,
                                             "external_field": args.ext_field})

    add_subtraction_results_to_dist_plots(subtraction_results, ax)

    add_results_to_dist_plot(results, ax, varied_gamma_pos=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$R$ (Å)')
    ax.set_ylabel(r'$\Delta \langle H \rangle$\textsubscript{GS} (eV)')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

    ax.legend(fancybox=False, edgecolor="k", framealpha=1, loc="lower right")

    ax.axhline(y=2.0 - 2, color='k', linewidth=1, linestyle='dashed', zorder=-100)
    ax.axvline(x=VdW_radius * 2, color='k', linewidth=1, linestyle='dashed', zorder=-100)
    ax.text(VdW_radius * 2.02, -1, r"I\textsubscript{2} vdW diameter", )

    fig.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.18)
    fig.savefig('spectrum.pdf')
    fig.show()


if __name__ == '__main__':
    main()
