import mpltex
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from plot_spectra.spectrum_two_oscillator import calc_truncated_energies_pair, calc_exact_energies
from utils import set_plot_style

set_plot_style(9)

linestyles = mpltex.linestyle_generator(lines=['-'], markers=[], hollow_styles=[True])

fig, ax1 = plt.subplots(figsize=(3.375, 2.5), constrained_layout=True)

left, bottom, width, height = [0.26, 0.45, 0.4, 0.4]  # h pos, v pos, h size, v size
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_xlim(1.8 - 0.01, 2.0 + 0.01)
ax2.set_ylim(1.4 + 0.01, 1.75 - 0.01)
# mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec="0.5")
# mark_inset(ax1, ax2, loc1=3, loc2=4, fc="none", ec="0.5")
mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec="0.5")

xmin, xmax = 0, 2
num_points = 2000
gammas = -np.linspace(xmin, xmax, num_points)
for (n, linestlye) in zip([4, 6, 8], ['.-', ':', '-']):
    print(n)
    energies = calc_truncated_energies_pair(gammas, n, {})[0]
    linestyle = next(linestyles)
    ax1.plot(abs(gammas), energies, **linestyle, label=r'$d=' + f'{int(2 ** (n / 2))}' + r'$', linewidth=1)
    ax2.plot(abs(gammas), energies, **linestyle, linewidth=1)

ax1.plot(abs(gammas), calc_exact_energies(0, 0, gammas), color='k', linestyle='--', label='Analytic', linewidth=1)
ax2.plot(abs(gammas), calc_exact_energies(0, 0, gammas), color='k', linestyle='--', linewidth=1)

ax1.set_xlabel(r'-$\gamma$')
ax1.set_ylabel(r'$E \; \left(\frac{1}{2}\hbar \omega\right)$')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
ax1.legend(fancybox=False, edgecolor="k", framealpha=1, loc="best", ncol=2)

ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

fig.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)

fig.show()
fig.savefig('spectrum_truncated.pdf')
