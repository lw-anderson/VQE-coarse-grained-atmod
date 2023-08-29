import argparse
import json
import os

import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from operators.hamiltonian import get_harmonic_hamiltonian
from plot_spectra.spectrum_two_oscillator import gamma_to_dist, calc_exact_energies, dist_to_gamma

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

plt.rcParams.update(tex_fonts)

location = '/Users/jamesfox/lewis/git/atomisticmodelling_poster/atomisticmodelling/output/2_oscillators_final_results'

results_dict = np.load(os.path.join(location, 'subtraction_outputs.npy'), allow_pickle=True)[()]
print(results_dict)

for h_power in [1]:

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(3.375, 5), constrained_layout=True,
    #                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.375, 2.75))

    dists, energies, stdevs = [], [], []
    experiment_linestyles = mpltex.linestyle_generator(markers=[])
    averages_linestyles = mpltex.linestyle_generator(colors=[], lines=[], markers=['o'])

    for key in reversed(list(results_dict.keys())):
        experiment_linestyle = next(experiment_linestyles)

        vals = results_dict[key]
        measurements = np.array(vals[h_power])[:200, 0]
        if key == "[-0.0]" or key == "[0.0]":
            label = r"$R = \infty$"
        else:
            label = r"$R = " + "{:.2f}".format(gamma_to_dist(vals[0])) + r"$\AA"
        ax1.plot(measurements, linewidth=1, **experiment_linestyle, label=label)
        dists.append(gamma_to_dist(vals[0]))
        energies.append(np.mean(measurements))
        stdevs.append(np.std(measurements) / np.sqrt(len(measurements)))

    ax1.set_xlabel('Measurement')
    ax1.set_ylabel(r'$\langle \hat{H} \rangle \; \left(\frac{1}{2}\hbar \omega\right)$')

    ax1.text(s=r'\textbf{(a)}', x=0.05, y=0.88, transform=ax1.transAxes)
    ax1.set_xlim(0, len(measurements))
    ax1.set_ylim(2.06, 2.84)
    # ax1.set_ylim(2.2, 2.85)
    fig1.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
    ax1.legend(fancybox=False, edgecolor="k", framealpha=1, ncol=2, loc="lower center")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

    av_linestyle = next(averages_linestyles)

    ################################################

    fig2, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(3.375, 2.75), sharex=True)

    ax2.set_ylabel(r'Mean $\langle \hat{H} \rangle$')
    ax3.set_ylabel(r'Exact $\langle \hat{H} \rangle$')
    ax4.set_ylabel(r'Error')

    xlims_broken_ax_left = (2.8, 6.2)
    xlims_broken_ax_right = (9.45, 10.55)

    size_ratio = (xlims_broken_ax_right[1] - xlims_broken_ax_right[0]) / (
            xlims_broken_ax_left[1] - xlims_broken_ax_left[0])

    # broken axes for ax2
    divider = make_axes_locatable(ax2)
    ax2_2 = divider.new_horizontal(size=f"{100 * size_ratio}%", pad=0.15)
    fig2.add_axes(ax2_2)
    ax2.set_xlim(xlims_broken_ax_left)
    ax2.spines['right'].set_visible(False)
    ax2_2.set_xlim(xlims_broken_ax_right)
    ax2_2.spines['left'].set_visible(False)

    # broken axes for ax3
    divider3 = make_axes_locatable(ax3)
    ax3_2 = divider3.new_horizontal(size=f"{100 * size_ratio}%", pad=0.15)
    fig2.add_axes(ax3_2)
    ax3.set_xlim(xlims_broken_ax_left)
    ax3.spines['right'].set_visible(False)
    ax3_2.set_xlim(xlims_broken_ax_right)
    ax3_2.spines['left'].set_visible(False)

    # broken axes for ax4
    divider4 = make_axes_locatable(ax4)
    ax4_2 = divider4.new_horizontal(size=f"{100 * size_ratio}%", pad=0.15)
    fig2.add_axes(ax4_2)
    ax4.set_xlim(xlims_broken_ax_left)
    ax4.spines['right'].set_visible(False)
    ax4_2.set_xlim(xlims_broken_ax_right)
    ax4_2.spines['left'].set_visible(False)

    ax2.text(s=r'\textbf{(b)}', x=0.05, y=0.75, transform=ax2.transAxes)
    ax3.text(s=r'\textbf{(c)}', x=0.05, y=0.75, transform=ax3.transAxes)
    ax4.text(s=r'\textbf{(d)}', x=0.05, y=0.75, transform=ax4.transAxes)

    ax4.set_xlabel(r'$R$ (\AA)', x=0.5 / (1 - size_ratio))

    for (ax, ax_2) in zip([ax2, ax3, ax4], [ax2_2, ax3_2, ax4_2]):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=False)

        ax_2.xaxis.set_minor_locator(AutoMinorLocator())
        ax_2.yaxis.set_minor_locator(AutoMinorLocator())
        ax_2.tick_params(which='both', direction='in', top=True, bottom=True, left=False, right=True, labelleft=False)

        ax_ticks = np.arange(3, 7, 1)
        ax_minor_ticks = np.arange(xlims_broken_ax_left[0], xlims_broken_ax_left[1], 0.2)
        ax_2_ticks = [10]
        ax_2_minor_ticks = np.arange(xlims_broken_ax_right[0] + 0.15, xlims_broken_ax_right[1] - 0.15, 0.2)
        ax.set_xticks(ax_ticks)
        ax_2.set_xticks(ax_2_ticks)
        ax_2.set_xticklabels([r"$\infty$"])
        ax.xaxis.set_minor_locator(ticker.FixedLocator(ax_minor_ticks))
        ax_2.xaxis.set_minor_locator(ticker.FixedLocator(ax_2_minor_ticks))

        dx = .025  # how big to make the diagonal lines in axes coordinates
        dy = 2 * dx
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', linewidth=1, clip_on=False)
        # ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - dx, 1 + dx), (1 - dy, 1 + dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (-dy, +dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (-1 - dy, -1 + dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (-2 - dy, -2 + dy), **kwargs)

        rescaled_dx = dx / size_ratio
        kwargs.update(transform=ax2_2.transAxes)  # switch to the right axes
        ax_2.plot((-rescaled_dx, +rescaled_dx), (1 - dy, 1 + dy), **kwargs)
        ax_2.plot((-rescaled_dx, +rescaled_dx), (-dy, +dy), **kwargs)
        ax_2.plot((-rescaled_dx, +rescaled_dx), (-1 - dy, -1 + dy), **kwargs)
        ax_2.plot((-rescaled_dx, +rescaled_dx), (-2 - dy, -2 + dy), **kwargs)

    # ax2_inset = fig.add_axes([0.75, 0.3, 0.2, 0.4])
    fig2.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
    fig2.set_tight_layout(True)
    fig2.subplots_adjust(hspace=0)
    # fig.savefig(os.path.join(location, 'subtraction_plots.pdf'))

    exact_energies = []
    for subdir in sorted(os.listdir(location), reverse=True):
        if os.path.isdir(os.path.join(location, subdir)):
            state = np.load(os.path.join(location, subdir, 'optimisation_output.npy'),
                            allow_pickle=True)[0]
            args = argparse.Namespace(**json.load(open(os.path.join(location, subdir, 'arguments.json'), "r")))
            gammas = args.gammas
            hamiltonian = get_harmonic_hamiltonian(2, 4, np.array([[0, gammas], [gammas, 0]]), 'bin', numpy_array=True)
            exact_energy = np.dot(state.conj().T, np.dot(hamiltonian, state))
            exact_energies.append(exact_energy)

    ax2.errorbar(dists, energies, stdevs, markersize=3, capsize=3, **av_linestyle)
    ax2_2.errorbar(dists, energies, stdevs, markersize=3, capsize=3, **av_linestyle)
    ax3.plot(dists, exact_energies, markersize=3, **av_linestyle)
    ax3_2.plot(dists, exact_energies, markersize=3, **av_linestyle)
    ax4.errorbar(dists, np.array(energies) - np.real(np.array(exact_energies)), stdevs, markersize=3, capsize=3,
                 **av_linestyle)
    ax4_2.errorbar(dists, np.array(energies) - np.real(np.array(exact_energies)), stdevs, markersize=3, capsize=3,
                   **av_linestyle)

    continuous_dists = np.linspace(3, 11, 1000)
    analytic_energies = calc_exact_energies(0, 0, [dist_to_gamma(dist) for dist in continuous_dists])

    ax3.plot(continuous_dists, analytic_energies, color='k', linewidth=1)
    ax3_2.plot(continuous_dists, analytic_energies, color='k', linewidth=1)

    ax2.set_ylim(2.42, 2.68)
    ax2_2.set_ylim(2.42, 2.68)
    ax2.set_yticks([2.5, 2.6])
    ax2_2.set_yticks([2.5, 2.6])

    ax3.set_ylim(1.77, 2.03)
    ax3_2.set_ylim(1.77, 2.03)
    ax3.set_yticks([1.8, 1.9, 2.0])
    ax3_2.set_yticks([1.8, 1.9, 2.0])

    ax4.set_ylim(0.58, 0.67)
    ax4_2.set_ylim(0.58, 0.67)

    fig1.savefig(os.path.join(location, 'subtraction_plots1.pdf'))
    fig2.savefig(os.path.join(location, 'subtraction_plots2.pdf'))

    fig1.show()
    fig2.show()
