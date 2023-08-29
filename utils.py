import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from qiskit.quantum_info import Statevector, partial_trace, entropy
from scipy.sparse.linalg import eigsh

from operators.hamiltonian import get_harmonic_hamiltonian

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 9
}


def plot_state(state, n, gamma, state_ref=None, min_ref=None, min_mean=None, var=None, save=True, inset=False,
               show_text=False):
    """
    Plot line plot of the real values of a wavefunction.
    """
    plt.rcParams.update(tex_fonts)
    grid = (2 ** n) * np.linspace(0, 1, 2 ** n, endpoint=False)

    fig, ax = plt.subplots()

    text = r'$\gamma = $' + str(gamma)

    if min_ref is not None and state_ref is not None:
        overlap = np.abs(np.inner(state, state_ref)) ** 2
        text = text + f'\noverlap = {round(overlap, 5)} \n$E_0$ = {round(min_ref, 5)}'
        plt.plot(grid, state_ref, marker='o', color='k',
                 label=f'Exact diagonalisation')

    if min_mean is not None and var is not None:
        text = text + f'\n$E$ = {round(min_mean, 5)}' + r'$\pm$' + f' {round(np.sqrt(var), 5)}'

    label_re = f"VQA output (Re)"
    label_im = f"VQA output (Im)"

    ax.plot(grid, np.real(state), marker='x', linestyle='--', color='r', label=label_re)
    ax.plot(grid, np.imag(state), marker='x', linestyle=':', color='r', label=label_im)

    ax.set_xlabel('Basis state')
    ax.set_ylabel('Amplitude')

    if show_text:
        ax.text(0.05, 0.95, text,
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, bbox={'facecolor': 'white'})

    plt.legend()
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:0" + str(n) + "b}"))
    ax.xaxis.set_ticks(np.arange(0, len(state), 1))
    plt.xticks(rotation=45)
    ax.axis([-0.1, len(state) + 0.1, -1.1, 1.1])
    ax.grid(True, 'major', 'y')

    if inset:
        left, bottom, width, height = [0.6, 0.2, 0.25, 0.25]
        ax2 = fig.add_axes([left, bottom, width, height])

        state = np.reshape(state, [2 ** int(n / 2), 2 ** int(n / 2)])

        ax2.imshow(np.real(state), cmap='RdBu_r', vmin=-1, vmax=1)

        ax2.set_xticks(range(2 ** int(n / 2)), minor=False)
        ax2.set_xticks(np.arange(0.5, 2 ** int(n / 2) - 1, 0.5), minor=True)
        ax2.set_yticks(range(2 ** int(n / 2)), minor=False)
        ax2.set_yticks(np.arange(0.5, 2 ** int(n / 2) - 1, 0.5), minor=True)
        ax2.xaxis.set_major_formatter(StrMethodFormatter("{x:0" + str(int(n / 2)) + "b}"))
        ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:0" + str(int(n / 2)) + "b}"))

        ax2.grid(which='minor', linewidth=1)
        ax2.set_ylabel("QDO 1 basis")
        ax2.set_xlabel("QDO 2 basis")

    if save and inset:
        plt.savefig('statevector_inset.png', bbox_inches='tight')
        plt.close(fig)
    elif save:
        plt.savefig('statevector.png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None


def plot_two_oscillator_state(state, n, gamma, minimum, save=True):
    """
    Plot heat map of the real values of the wavefunction within the two oscillator Fock basis.
    """
    plt.rcParams.update(tex_fonts)
    state = np.reshape([abs(val) for val in state], [2 ** int(n / 2), 2 ** int(n / 2)])

    fig, ax = plt.subplots()
    col = ax.imshow(state, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(2 ** int(n / 2)), minor=False)
    ax.set_xticks(np.arange(0.5, 2 ** int(n / 2) - 1, 0.5), minor=True)
    ax.set_yticks(range(2 ** int(n / 2)), minor=False)
    ax.set_yticks(np.arange(0.5, 2 ** int(n / 2) - 1, 0.5), minor=True)
    ax.grid(which='minor', color='w', linewidth=1)
    fig.colorbar(col)
    plt.title(r"$\gamma = $" + str(gamma) + r"$E = $" + str(round(minimum, 5)))
    plt.ylabel("Oscillator 1 Fock state")
    plt.xlabel("Oscillator 2 Fock state")

    for i in range(2 ** int(n / 2)):
        for j in range(2 ** int(n / 2)):
            ax.text(j, i, '%.4f' % state[i, j], ha="center", va="center", color="k")

    if save:
        plt.savefig('two_oscillator_state.png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return None


def plot_cost_function_progress(cost_function_values, gamma, exact_cost_function_values=None, min_analytic=None,
                                min_truncated=None, min_mean=None, var=None, max_steps=None, ylims=None,
                                save=True,
                                eval_routine=None, show_text=False, x_values=None):
    """
    Plot values of cost function evaluated during optimisation process
    """

    mean_values = np.array(cost_function_values)[:, 0]
    variance_values = np.array(cost_function_values)[:, 1] ** 2

    plt.rcParams.update(tex_fonts)
    fig, ax1 = plt.subplots()

    if min_truncated is not None:
        ax1.set_ylim(min_truncated - 0.5, min_truncated + 2.5)

    if ylims is not None:
        left, bottom, width, height = [0.4, 0.4, 0.4, 0.4]  # h pos, v pos, h size, v size
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.set_ylim(ylims[0], ylims[1])
        ax2.set_ylabel(r"$E$")
        mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec="0.5")
        mark_inset(ax1, ax2, loc1=3, loc2=4, fc="none", ec="0.5")

        ax2.plot(mean_values, 'k', linewidth=1, alpha=0.8)

        fname = 'evaluations_zoom.png'

        axes = [ax1, ax2]

        rolling_variance_steps = 50
        rolling_variances = []
        converged_point = None
        for i, estimated_var in enumerate(variance_values):
            rolling_var = np.var(mean_values[max(0, i - rolling_variance_steps): max(0, i)])
            rolling_variances.append(rolling_var)

            if rolling_var < estimated_var and converged_point is None and i > rolling_variance_steps:
                converged_point = i
                ax1.vlines([converged_point, converged_point - rolling_variance_steps], 0, 1,
                           transform=ax1.get_xaxis_transform(), linestyles='dashed', color='k')
                ax2.vlines([converged_point, converged_point - rolling_variance_steps], 0, 1,
                           transform=ax2.get_xaxis_transform(), linestyles='dashed', color='k')

        ax3 = ax2.twinx()
        ax3.set_yscale('log')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.plot(rolling_variances, color='r', linestyle='-', linewidth=1)
        ax3.plot(variance_values, color='r', linestyle='--', linewidth=1)
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_ylabel('Std Dev', color='r')

    else:
        axes = [ax1]
        fname = 'evaluations.png'
        if x_values is not None:
            ax2 = ax1.twinx()
            for i, x in enumerate(np.array(x_values).T):
                ax2.plot(x, linestyle='--', color='r', linewidth=1)
                ax2.text(len(x_values[:, 0]) + 1, x[-1] + 0.01, r'$\theta_{' + str(i) + '}$', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.set_ylabel('Parameter value', color='r')

    if max_steps is None:
        mean_values = mean_values
    else:
        mean_values = mean_values[:max_steps]

    txt = r"$\gamma =$" + f"{gamma}"

    for ax in axes:
        if min_analytic is not None:
            txt = txt + "\n" + r"$E_{analytic} = $" + str(round(min_analytic, 5))
            ax.axhline(min_analytic, 0, 1, linestyle='dashed', color='k', linewidth=1)
        if min_truncated is not None:
            ax.axhline(min_truncated, 0, 1, linestyle='dashed', color='k', linewidth=1)
            txt = txt + "\n" + r"$E_{truncated} = $" + str(round(min_truncated, 5))
        if min_mean is not None and var is not None:
            txt = txt + "\n" + r"$E_{optimised} = $" + str(round(min_mean, 5)) + r"$\pm$" + str(
                round(np.sqrt(var), 5))
        elif min_mean is not None:
            txt = txt + "\n" + r"$E_{optimised} = $" + str(round(min_mean, 5))

        ax.plot(mean_values, 'k', linewidth=1, alpha=0.8)

        if exact_cost_function_values is not None:
            ax.plot(exact_cost_function_values, 'c', linewidth=1)

        if show_text:
            ax.text(0.05, 0.95, txt,
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, bbox={'facecolor': 'white'})

        if eval_routine is not None and len(eval_routine) > 0:
            ax.vlines(eval_routine[:-1], 0, 1, transform=ax.get_xaxis_transform(), linestyles='dashed', color='k')

        if max_steps is None:
            ax.set_xlim(0, len(cost_function_values))
        else:
            ax.set_xlim(0, max_steps)

    ax1.set_xlabel("Optimisation step")
    ax1.set_ylabel(r"$E$")

    if save:
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return None


def plot_repeated_measurement_distribution(measured_energies, predicted_variances, noiseless_mean=None, text=""):
    """
    Find mean and variance and plot distribution for cost function evaluated multiple times.
    """

    mean = np.mean(measured_energies)
    variance = np.var(measured_energies)
    num_evaluations = len(measured_energies)
    fig, ax = plt.subplots()
    plt.hist(measured_energies)
    text = text + f'\nmean = {mean}\nvariance = {variance}'

    if noiseless_mean is not None:
        text = text + f'\n no noise = {noiseless_mean}'
        ax.axvline(noiseless_mean, 0, 1, linestyle='dashed', color='k', linewidth=1)

    ax.text(0.05, 0.95, text,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, bbox={'facecolor': 'white'})

    print(f'Mean: {mean}')
    print(f'Measured standard deviation for single eval: {np.sqrt(variance)}')
    print(f'Measured standard deviation for repeated final result: {np.sqrt(variance / num_evaluations)}')
    print(f'Estimated standard deviation: {np.sqrt(np.mean(predicted_variances))}')

    print('====================')
    print('Estimated variance ', np.mean(predicted_variances))
    print('Measured variance ', np.var(measured_energies))
    print('====================')

    fname = 'noise_estimate.png'

    plt.savefig(fname)
    return mean, variance, variance / num_evaluations


def produce_all_plots(args, state, min_optimised, state_reference, min_truncated, cost_function_values,
                      exact_cost_function_values, x_values, min_analytic, ylims, save, eval_routine=[]):
    n = args.num_qubits
    gammas = args.gammas
    if n <= 4:
        plot_state(state, n, gammas, state_ref=state_reference, min_ref=min_truncated, save=save, show_text=True)
        plot_state(state, n, gammas, state_ref=state_reference, min_ref=min_truncated, save=save, inset=True)
        plot_two_oscillator_state(state, n, gammas, minimum=min_optimised, save=save)
    plot_cost_function_progress(cost_function_values=cost_function_values, gamma=gammas,
                                exact_cost_function_values=exact_cost_function_values,
                                min_truncated=min_truncated, min_analytic=min_analytic,
                                save=save, eval_routine=eval_routine, x_values=x_values)
    plot_cost_function_progress(cost_function_values=cost_function_values, gamma=gammas,
                                exact_cost_function_values=exact_cost_function_values,
                                min_truncated=min_truncated, min_analytic=min_analytic,
                                ylims=ylims, save=save, eval_routine=eval_routine,
                                x_values=x_values)
    plt.close('all')


def create_output_path(args, version=0):
    anharmonic = True if args.cubic or args.quartic or args.ext_field else False
    output_path = os.path.join(args.save_location,
                               f'qhos_{args.num_oscillators}_enc_{args.encoding}_qubits-{args.num_qubits}'
                               f'_anharmonic_{anharmonic}'
                               f'_ansatz-{args.ansatz}_depth-{args.depth}'
                               f'_backend-{args.backend}_shots-{args.shots}'
                               f'_noise-{args.noise}_solver-{args.solver}_v{version}')
    if os.path.exists(output_path):
        return create_output_path(args, version=version + 1)
    else:
        return output_path


def create_json_filename(version=0):
    if version == 0:
        fname = 'arguments.json'
    else:
        fname = f'arguments_{version}.json'
    if os.path.exists(fname):
        return create_json_filename(version=version + 1)
    else:
        return fname


def combine_runtime_args(args):
    """
    Takes args with optional argument for output directory. If output directory exists, loads runtime arguments from
    output directory and modifies args to contain arguments used when running optimisation.
    """

    if args.directory is not None:
        os.chdir(args.directory)
        runtime_args = json.load(open("arguments.json", "r"))
        runtime_args = argparse.Namespace(**runtime_args)

        if os.path.isfile("costfunc_values.txt") and os.path.isfile("parameters_values.txt"):
            cost_function_values = np.loadtxt("costfunc_values.txt")
            x_values = np.loadtxt("parameters_values.txt")

        elif os.path.isfile("MIDACO_HISTORY.TXT"):
            midaco_output = np.loadtxt("MIDACO_HISTORY.TXT", skiprows=12)
            cost_function_values = midaco_output[:, 0]
            x_values = midaco_output[:, 1:]

        else:
            raise FileNotFoundError("Could not find output file.")

        (minimum_optimised, position) = min((val, idx) for (idx, val) in enumerate(cost_function_values))
        args.parameters = x_values[position, :]

        args.cost_func = runtime_args.cost_func
        args.ansatz = runtime_args.ansatz
        args.gamma = runtime_args.gamma
        args.gamma12 = runtime_args.gamma12
        args.gamma23 = runtime_args.gamma23
        args.gamma31 = runtime_args.gamma31
        args.shots = runtime_args.shots
        args.num_qubits = runtime_args.num_qubits
        args.depth = runtime_args.depth
        args.noise = runtime_args.noise
        args.ants = runtime_args.ants
        args.kernel = runtime_args.kernel

    else:
        # turn string input argument into list
        for pattern in [' ', '[', ']']:
            args.parameters = args.parameters.replace(pattern, '')
        args.parameters = [float(param) for param in args.parameters.split(',')]

    args.solver = 'nlopt'

    return args


def entanglement_entropy(state, qubits_per_oscillator, num_oscillators, oscillator):
    """
    Calculate entanglement entropy using subsystem of single oscillator.
        S = -Tr[ρ_A ln ρ_A]
    where ρ_A is state with all oscillators except A traced out.
    """

    if type(qubits_per_oscillator) is not int or type(num_oscillators) is not int:
        raise ValueError('qubits_per_oscillator and num_oscillators must be ints.')

    if len(state) != 2 ** (qubits_per_oscillator * num_oscillators):
        raise ValueError('Size mismatch with state and number of qubits required.')

    sv = Statevector(state, [2 ** qubits_per_oscillator, ] * num_oscillators)

    subsystems = list(range(num_oscillators))
    if oscillator not in subsystems:
        raise ValueError('oscillator must be int in [0, num_oscillators)')
    subsystems.remove(oscillator)

    rho_oscillator = partial_trace(sv, qargs=subsystems)

    return entropy(rho_oscillator)


def get_qubit_mapping_from_transpiled_circuit(qc):
    if qc._layout:
        qubit_dict = qc._layout.get_virtual_bits()

        dict_no_ancilla = {phys_qubit: virt_qubit for phys_qubit, virt_qubit in qubit_dict.items()
                           if phys_qubit.register.name != 'ancilla'}

        return dict_no_ancilla
    else:
        return {qubit: i for (i, qubit) in enumerate(qc.qregs[0])}


def gammas_list_to_matrix(gammas):
    num_oscillators = (np.sqrt(8 * len(gammas) + 1) + 1) / 2

    if num_oscillators % 1 != 0.0:
        raise ValueError("length of gammas list must be n(n-1)/2 wherre n is int.")
    num_oscillators = int(num_oscillators)
    gammas_matrix = np.zeros((num_oscillators, num_oscillators))
    gammas_matrix[np.triu_indices(num_oscillators, 1)] = gammas
    gammas_matrix += gammas_matrix.transpose()
    return gammas_matrix


def truncated_energies(gammas, qubits_per_oscillator):
    num_oscillators = gammas.shape[0]
    hamiltonian = get_harmonic_hamiltonian(num_oscillators, num_oscillators * qubits_per_oscillator, gammas,
                                           encoding='bin')
    eval, evec = eigsh(hamiltonian, k=2, which="SA")
    evec = evec.transpose()
    ev_list = [tup for tup in zip(eval, evec)]
    ev_list.sort(key=lambda tup: tup[0], reverse=False)
    eval, evec = zip(*ev_list)
    return eval


def calc_truncated_energies(varying_gammas, qubits_per_oscillator):
    eigenvalues = np.array([truncated_energies(gammas, qubits_per_oscillator) for gammas in varying_gammas])
    return eigenvalues.transpose()


def truncated_energies_pair(gamma, num_qubits):
    hamiltonian = get_harmonic_hamiltonian(2, num_qubits, np.array([[0, gamma], [gamma, 0]]), encoding='bin')
    eval, evec = eigsh(hamiltonian, k=2, which="SA")
    ev_list = [tup for tup in zip(eval, evec)]
    ev_list.sort(key=lambda tup: tup[0], reverse=False)
    eval, evec = zip(*ev_list)
    return eval


def set_plot_style(fontsize=9):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "axes.titlesize": fontsize
    }
    plt.rcParams.update(tex_fonts)

    return
