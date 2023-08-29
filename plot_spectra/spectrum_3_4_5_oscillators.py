import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from calculate_gammas import get_polygon_gammas
from collect_results import add_results_to_plot, collect_results
from simplectic_ground_state import exact_ground_state_energy
from utils import set_plot_style

set_plot_style()


def main():
    qubits_per_oscillator = 2
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.75))  # , gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})

    # fig.subplots_adjust(hspace=0.0)

    dir_3_qhos = "/home/lewis/git/atomisticmodelling/results/many_oscillator_results_no_repeats/qhos_3"
    dir_4_qhos = "/home/lewis/git/atomisticmodelling/results/many_oscillator_results_no_repeats/qhos_4"
    dir_5_qhos = "/home/lewis/git/atomisticmodelling/results/many_oscillator_results_no_repeats/qhos_5"

    for (num_oscillators, directory, energy_ax) in zip([3, 4, 5], [dir_3_qhos, dir_4_qhos, dir_5_qhos],
                                                       axes):
        gammas = np.linspace(-0.1, 2, 1000)  # was -0.1 for first arg
        all_gamma_matrices = [get_polygon_gammas(gamma, num_oscillators) for gamma in gammas]

        add_simplectic_full_energy_to_plot(gammas, all_gamma_matrices, energy_ax)

        add_results_to_plot(collect_results(directory, all_plots=False), energy_ax)

        if num_oscillators == 2:
            energy_ax.set_xlim(-0.05, 1.2)
            energy_ax.set_ylim(1.4, 1.2)
        elif num_oscillators == 3:
            energy_ax.set_xlim(-0.05, 1.25)
            energy_ax.set_ylim(2.3, 3.05)
        elif num_oscillators == 4:
            energy_ax.set_xlim(-0.02, 0.42)
            energy_ax.set_ylim(3.55, 4.05)
        elif num_oscillators == 5:
            energy_ax.set_xlim(-0.01, 0.261)
            energy_ax.set_ylim(4.3, 5.05)

        energy_ax.xaxis.set_minor_locator(AutoMinorLocator())
        energy_ax.yaxis.set_minor_locator(AutoMinorLocator())
        energy_ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
        energy_ax.set_xlabel(r'$2\alpha/D^3$')

        inset_ax = inset_axes(energy_ax, width=0.6, height=0.6, loc="lower left")

        plot_geometry(num_oscillators, inset_ax)

    axes[0].set_ylabel(r'$E\;\left(\frac{1}{2}\hbar \omega\right)$')
    # axes[1, 0].set_ylabel(r'error $\;\left(\frac{1}{2}\hbar \omega\right)$')

    for (ax, subplot) in zip(axes, ["a", "b", "c"]):
        ax.text(s=r'\textbf{(' + subplot + ')}', x=0.85, y=0.85, transform=ax.transAxes)

    fig.subplots_adjust(bottom=0.45, left=0.07, right=0.93)
    legend = axes[0].legend(fancybox=False, edgecolor="k", framealpha=1, ncol=3,
                            loc="lower center", bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure)
    legend.get_frame().set_linewidth(0.75)
    fig.tight_layout()
    fig.show()

    fig.savefig("3_4_5_oscillators_spectrum.pdf")


def plot_geometry(num_oscillators, ax, geometry="regular polygon"):
    ax.set_ylim(-1.2, 1.5)
    ax.set_xlim(-1.35, 1.35)
    ax.set_axis_off()

    thetas = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(thetas), np.sin(thetas), 'k--', linewidth=0.5)

    oscillators_x = np.cos(np.linspace(np.pi / 2, 5 * np.pi / 2, num_oscillators + 1))
    oscillators_y = np.sin(np.linspace(np.pi / 2, 5 * np.pi / 2, num_oscillators + 1))

    for i in range(num_oscillators):
        if geometry == "regular polygon":
            for j in range(num_oscillators):
                ax.plot([oscillators_x[i], oscillators_x[j]],
                        [oscillators_y[i], oscillators_y[j]],
                        'k-', linewidth=0.5)
        elif geometry == "ring":
            ax.plot([oscillators_x[i], oscillators_x[(i + 1) % num_oscillators]],
                    [oscillators_y[i], oscillators_y[(i + 1) % num_oscillators]],
                    'k-', linewidth=0.5)
        else:
            raise ValueError("geometry must be one ore regular poylgon or ring")

    ax.plot(oscillators_x, oscillators_y, markersize=4, markerfacecolor='white', markeredgecolor="k", marker='o',
            linewidth=0, markeredgewidth=0.5)

    if num_oscillators == 3:
        ax.text(0, 1.2, r"$D$", ha="center", va="bottom")
        ax.arrow(0, 1.21, 1, 0, head_width=0.075, head_length=0.05, linewidth=0.5, color='k',
                 length_includes_head=True, linestyle='-')
        ax.arrow(0, 1.21, -1, 0, head_width=0.075, head_length=0.05, linewidth=0.5, color='k',
                 length_includes_head=True, linestyle='-')


if __name__ == '__main__':
    main()


def add_simplectic_full_energy_to_plot(varied_gamma_values, all_gamma_values, ax):
    ground_state_energies = [exact_ground_state_energy(gamma_matrix) for gamma_matrix in all_gamma_values]
    kwargs = {'color': 'k', 'linewidth': 1, 'linestyle': '-'}
    ax.plot(varied_gamma_values, ground_state_energies, **kwargs)
    ax.plot([], [], **kwargs, label='Analytic')

    return ground_state_energies


def add_simplectic_two_body_energy_to_plot(varied_gamma_values, all_gamma_values, ax):
    num_oscillators = all_gamma_values[0].shape[0]
    energies = []
    for gamma_matrix in all_gamma_values:
        energy = 0.0
        for i in range(0, num_oscillators):
            for j in range(i + 1, num_oscillators):
                energy += exact_ground_state_energy(np.array([[0, gamma_matrix[i][j]], [gamma_matrix[i][j], 0]]))
        if num_oscillators == 2:
            pass
        elif num_oscillators == 3:
            energy -= 3
        elif num_oscillators == 4:
            energy -= 8
        elif num_oscillators == 5:
            energy -= 15
        else:
            pass  # raise ValueError('num_oscillators must be 3,4 or 5.')

        energies.append(energy)

    kwargs = {'color': 'k', 'linestyle': '--'}
    ax.plot(varied_gamma_values, energies, **kwargs)
    ax.plot([], [], **kwargs, label='QDOs, two body only')

    return energies
