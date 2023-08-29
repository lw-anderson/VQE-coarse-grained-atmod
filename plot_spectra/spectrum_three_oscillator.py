import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

from collect_results import add_results_to_plot
from operators.hamiltonian import get_harmonic_hamiltonian
from plot_spectra.spectrum_two_oscillator import truncated_energies_pair
from simplectic_ground_state import three_oscillator_exact_ground_state_energy, \
    two_oscillator_exact_ground_state_energy
from utils import set_plot_style

set_plot_style()

parser = argparse.ArgumentParser(description='Load outputs of optimisation routine and produce plots.')
parser.add_argument('--results',
                    default='',
                    type=str, help='Directory containing all results to add to spectrum.')
parser.add_argument('--fixed-gamma', type=float, default=None,
                    help='If not None, will fixed two of the gammas to this values')
parser.add_argument('--save-fig', action='store_true', help='Flag to save plots')
parser.add_argument('--save-spectra', action='store_true', help='Save spectra calculated. Useful for ARCUS jobs.')
parser.add_argument('--load', action='store_true', help='Load saved models and add to plot.')
args = parser.parse_args()


def truncated_energies_triple(gamma12, gamma23, gamma31, n):
    gammas = np.zeros((3, 3))
    gammas[np.triu_indices(3, 1)] = [gamma12, gamma23, gamma31]
    gammas[np.tril_indices(3, -1)] = [gamma12, gamma23, gamma31]
    hamiltonian = np.float64(get_harmonic_hamiltonian(3, n, gammas, encoding='bin'))
    eval, evec = eig(hamiltonian)
    evec = evec.transpose()
    ev_list = [tup for tup in zip(eval, evec)]
    ev_list.sort(key=lambda tup: tup[0], reverse=False)
    eval, evec = zip(*ev_list)
    # print(rerwite_state(np.sign(evec[0][0]) * evec[0]))
    return eval


def calc_truncated_energies_triple(gammas, n):
    eigenvalues = np.array(
        [truncated_energies_triple(gamma12, gamma23, gamma31, n) for (gamma12, gamma23, gamma31) in gammas])
    return eigenvalues.transpose()


def calc_truncated_energies_pair(gammas, n):
    eigenvalues = np.array([truncated_energies_pair(gamma, n) for gamma in gammas])
    return eigenvalues.transpose()


def add_truncated_energies_to_plot(varied_gamma_values, all_gamma_values, n, ax):
    if n == 6:
        kwargs1 = {'color': 'r'}
        kwargs2 = {'color': 'b'}
        ax.plot([], [], label=f'{n} qubits', **kwargs1)
        ax.plot([], [], label=f'{n} qubits, two body only', **kwargs2)
    else:
        kwargs1 = {'color': 'r', 'linestyle': '--'}
        kwargs2 = {'color': 'b', 'linestyle': '--'}

    truncated_spectrum = calc_truncated_energies_triple(all_gamma_values, n)
    for energies in truncated_spectrum:
        ax.plot(varied_gamma_values, energies, **kwargs1)

    two_oscillator_spectrum_1 = calc_truncated_energies_pair(np.array(all_gamma_values)[:, 0], int(2 * n / 3))
    two_oscillator_spectrum_2 = calc_truncated_energies_pair(np.array(all_gamma_values)[:, 1], int(2 * n / 3))
    two_oscillator_spectrum_3 = calc_truncated_energies_pair(np.array(all_gamma_values)[:, 2], int(2 * n / 3))

    two_body_energies = two_oscillator_spectrum_1[0] + two_oscillator_spectrum_2[0] + two_oscillator_spectrum_3[0] - 3
    ax.plot(varied_gamma_values, two_body_energies, **kwargs2)

    if args.save_spectra:
        np.save(f'n={n}_gammas={all_gamma_values[0]}_threebody', [varied_gamma_values, truncated_spectrum])
        np.save(f'n={n}_gammas={all_gamma_values[0]}_twobody', [varied_gamma_values, two_body_energies])

    return truncated_spectrum, two_body_energies


def analytic_pair(gamma):
    return np.sqrt(1 + gamma / 2) + np.sqrt(1 - gamma / 2)


def all_analytic_pairs(gamma12, gamma23, gamma31):
    return analytic_pair(gamma12) + analytic_pair(gamma23) + analytic_pair(gamma31) - 3


def add_analytic_pairs_to_plot(varied_gamma_values, all_gamma_values, ax):
    two_body_energies = [all_analytic_pairs(gamma12, gamma23, gamma31)
                         for (gamma12, gamma23, gamma31) in all_gamma_values]
    kwargs = {'color': 'k', 'linestyle': 'dashdot'}
    ax.plot(varied_gamma_values, two_body_energies, **kwargs)
    ax.plot([], [], **kwargs, label='Analytic two body only')

    return None


def add_simplectic_three_body_energy_to_plot(varied_gamma_values, all_gamma_values, ax):
    three_body_energies = [three_oscillator_exact_ground_state_energy(gamma12, gamma23, gamma31)
                           for (gamma12, gamma23, gamma31) in all_gamma_values]
    kwargs = {'color': 'k'}
    ax.plot(varied_gamma_values, three_body_energies, **kwargs)
    ax.plot([], [], **kwargs, label='Exact')

    return None


def add_simplectic_two_body_energy_to_plot(varied_gamma_values, all_gamma_values, ax):
    three_body_energies = [two_oscillator_exact_ground_state_energy(gamma12)
                           + two_oscillator_exact_ground_state_energy(gamma23)
                           + two_oscillator_exact_ground_state_energy(gamma31) - 3
                           for (gamma12, gamma23, gamma31) in all_gamma_values]
    kwargs = {'color': 'k', 'linestyle': '--'}
    ax.plot(varied_gamma_values, three_body_energies, **kwargs)
    ax.plot([], [], **kwargs, label='Exact, two body only')

    return None


def rerwite_state(vec, n=6):
    grid = (2 ** n) * np.linspace(0, 1, 2 ** n, endpoint=False)
    amplitudes = []
    fock_states = []
    for amplitude, basis in zip(vec, grid):
        string_format = "{0:0" + str(n) + "b}"
        basis = string_format.format(int(basis))
        fock_state_1 = int(basis[0:int(n / 3)], 2)
        fock_state_2 = int(str(basis[int(n / 3):int(2 * n / 3)]), 2)
        fock_state_3 = int(basis[int(2 * n / 3):int(n)], 2)
        amplitudes.append(amplitude)
        fock_states.append((fock_state_1, fock_state_2, fock_state_3))

    paired_list = [tup for tup in zip(amplitudes, fock_states)]
    paired_list.sort(key=lambda tup: abs(tup[0]), reverse=True)
    return (paired_list)


def plot_required_fidelity(varied_gamma_values, truncated_spectrum, two_body_energies, ax):
    truncated_spectrum = np.array(truncated_spectrum)
    two_body_energies = np.array(two_body_energies)
    # delta = two_body_energies[0]-truncated_spectrum[0]
    two_body_eigen_val_0 = two_body_energies[0]
    eigen_val_0 = truncated_spectrum[0]
    eigen_val_1 = truncated_spectrum[1]
    eigen_val_top = truncated_spectrum[-1]

    for factor in [1.0, 0.1]:
        max_fidelities = np.sqrt(1 - abs(factor * (eigen_val_0 - two_body_eigen_val_0) / (eigen_val_1 - eigen_val_0)))
        min_fidelities = np.sqrt(1 - abs(factor * (eigen_val_0 - two_body_eigen_val_0) / (eigen_val_top - eigen_val_0)))
        ax.plot(varied_gamma_values, 1 - np.nan_to_num(min_fidelities), 'k', linewidth=0.5)
        ax.plot(varied_gamma_values, 1 - np.nan_to_num(max_fidelities), 'k', linewidth=0.5)

        if factor == 1:
            ax.fill_between(varied_gamma_values, 1 - np.nan_to_num(min_fidelities), 1 - np.nan_to_num(max_fidelities),
                            interpolate=True, facecolor='black', alpha=0.25,
                            label=r'Error $<$ $E_{\textrm{(3 body)}}$', hatch='\\')
        elif factor == 0.1:
            ax.fill_between(varied_gamma_values, 1 - np.nan_to_num(min_fidelities), 1 - np.nan_to_num(max_fidelities),
                            interpolate=True, facecolor='black', alpha=0.5,
                            label=r'Error $<$ 10\% $E_{\textrm{(3 body)}}$', hatch='//')


def main():
    gammas = np.linspace(0, 2, 50)

    if args.fixed_gamma is None:
        # Use all gammas equal
        all_three_gammas = [(gamma, gamma, gamma) for gamma in gammas]
    else:
        # Fix two gammas
        all_three_gammas = [(args.fixed_gamma, gamma, args.fixed_gamma) for gamma in gammas]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))

    qubits_per_oscillator = 2
    n = 3 * qubits_per_oscillator
    truncated_spectrum, two_body_energies = add_truncated_energies_to_plot(gammas, all_three_gammas, n, ax1)

    add_simplectic_three_body_energy_to_plot(gammas, all_three_gammas, ax1)
    add_simplectic_two_body_energy_to_plot(gammas, all_three_gammas, ax1)

    ax1.set_xlim(0, 2)
    ax1.set_ylim(2.1, 3.1)
    ax1.set_ylabel(r'$E$')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # add_truncated_energies_to_plot(gammas, all_three_gammas, 9, ax1)

    plot_required_fidelity(gammas, truncated_spectrum, two_body_energies, ax2)

    if args.results:
        print(args.results)
        add_results_to_plot(args.read_output(args.results), ax1, ax2, varied_gamma_pos=1)

    ax1.legend(loc='lower right')

    ax2.legend(loc='lower right')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$1-F$')
    ax2.set_xlabel(r'varied $\gamma$')
    ax2.set_ylim(1e-5, 1)

    plt.subplots_adjust(hspace=0)
    if args.save_fig:
        if args.fixed_gamma is None:
            plt.savefig(f'Three_body_spectrum_fixed_gamma_{args.fixed_gamma}.png')
        else:
            plt.savefig(f'Three_body_spectrum')
    plt.show()


if __name__ == '__main__':
    main()
