import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_results import collect_results
from plot_spectra.spectrum_two_oscillator_anharmonic import calculate_energy, \
    one_oscillator_ext_field
from utils import set_plot_style

set_plot_style()

figsize = (3.375, 2.5)

linestyles = mpltex.linestyle_generator(lines=[], markers=['o', 's', 'v'], hollow_styles=[True])
linestyles_sub = mpltex.linestyle_generator(lines=[], markers=['o', 's', 'v'], hollow_styles=[False])

plt.rcParams.update(tex_fonts)
fig, ax = plt.subplots(figsize=figsize)
ax.set_ylabel(r'$\Delta\langle H \rangle$\textsubscript{GS} $\; \left(\frac{1}{2}\hbar \omega\right)$')
ax.set_xlabel(r'$E$')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
ax.set_xlim(-0.02, 0.52)

results_directory = '/home/lewis/git/atomisticmodelling/results/1_oscillator_real_device'
results = collect_results(results_directory)
kwargs = {"quartic_prefactor": 0.2}
ax.set_ylim(- 0.14, 0.09)

fields = np.linspace(-1, 1, 1000)
anharm_energies = [calculate_energy(one_oscillator_ext_field(x, log_d=2, **kwargs)) for x in fields]
harm_energies = [calculate_energy(one_oscillator_ext_field(x, log_d=2)) for x in fields]
anharm_offset = calculate_energy(one_oscillator_ext_field(0.0, **kwargs))
harm_offset = calculate_energy(one_oscillator_ext_field(0.0))

ax.plot(fields, np.array(anharm_energies) - anharm_offset, color='k', linewidth=1, label='Anharmonic analytic',
        zorder=-32)
ax.plot(fields, np.array(harm_energies) - harm_offset, color='k', linestyle='--', linewidth=1,
        label='Harmonic analytic', zorder=-32)

# which points to plot
linestyles = mpltex.linestyle_generator(lines=[], markers=['o', 's', 'v'], hollow_styles=[False])
linestyle_extrap_lanczos = next(linestyles)
linestyle_exact = next(linestyles)
linestyle_exp = None  # next(linestyles)
linestyle_lanczos_extrap = next(linestyles)

labelled = False
for label in results:
    for (varied_hamiltonian_param, minimum_mean, mean_repeated_evals_stdev, lanczos_mean, lanczos_stdev,
         extrap_then_lanczos_mean, extrap_then_lanczos_stdev,
         lanczos_then_extrap_mean, lanczos_then_extrap_stdev,
         subtracted_mean, subtracted_stdev,
         overlap, exact_final_output) in results[label]:
        ax.plot([varied_hamiltonian_param[0] if varied_hamiltonian_param is list
                 else varied_hamiltonian_param],
                [exact_final_output[0] - anharm_offset],
                **linestyle_exact, markersize=4, zorder=-20, fillstyle="none")
        # ax.errorbar([varied_hamiltonian_param[0] if varied_hamiltonian_param is list
        #              else varied_hamiltonian_param],
        #             [lanczos_then_extrap_mean - anharm_offset],
        #             [extrap_then_lanczos_stdev],
        #             **linestyle_lanczos_extrap, capsize=3, markersize=4)
        if not labelled:
            ax.errorbar([varied_hamiltonian_param[0] if varied_hamiltonian_param is list
                         else varied_hamiltonian_param],
                        [extrap_then_lanczos_mean - anharm_offset],
                        [extrap_then_lanczos_stdev],
                        **linestyle_extrap_lanczos, capsize=3, markersize=4,
                        label=r'\textit{ibmq\_montreal} + mitigation')
            labelled = True
        else:
            ax.errorbar([varied_hamiltonian_param[0] if varied_hamiltonian_param is list
                         else varied_hamiltonian_param],
                        [extrap_then_lanczos_mean - anharm_offset],
                        [extrap_then_lanczos_stdev],
                        **linestyle_extrap_lanczos, capsize=3, markersize=4)

ax.plot([], [], **linestyle_exact, markersize=4, label='Noise free evaluation', fillstyle="none")
# ax.plot([], [], **linestyle_exp, label=r'\textit{ibm_montreal} (simulated)')
# ax.errorbar([], [], [], **linestyle_extrap_lanczos, capsize=3, markersize=4, label=r'AAA')
# ax.errorbar([], [], [], capsize=10, markersize=4, label=r'\textit{ibmq\_montreal} + mitigation')
ax.legend(fancybox=False, edgecolor="k", framealpha=1, loc="lower left")
fig.tight_layout()
fig.savefig("extended_spectra/spectrum_extended_one_oscillator_"
            + str(kwargs["quartic_prefactor"])
            + "_quartic_simulated.pdf")
fig.show()
