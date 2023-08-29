import mpltex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase


def delta_E(gam):
    """
    1D London dispersion in units of (1/2) hbar omega
    """
    return 0.1 * (-1 / 16) * gam ** 2


def gamma(alpha, r):
    """
    Calculate gam given polarisabilty alpha in A^3 and separation r in A.
    """
    return 4 * alpha * r ** 3


def shots_grouped_uniform_measure(N, d, a, r):
    gam = gamma(a, r)
    epsilon = delta_E(gam)
    term_1 = np.sqrt(2) / 3 * np.sqrt(N) * np.sqrt(d ** 2 - 1)
    if N % 2 == 0:
        term_2a = abs(gam) * np.sqrt(N / 2) * (N - 1)
    else:
        term_2a = abs(gam) * np.sqrt((N - 1) / 2) * N

    term_2b = (2 * d - (4 + 2 * np.log2(d)) * d ** 2 + (2 + 2 * np.log2(d) + (1 / 2) * np.log2(d) ** 2))
    return (1 / epsilon ** 2) * (term_1 + term_2a * term_2b) ** 2


def shots_ungrouped_uniform_measure(N, d, a, r):
    gam = gamma(a, r)
    epsilon = delta_E(gam)
    term_1 = np.sqrt(6) / 4 * N * (d ** 2 - 1)
    term_2a = abs(gam) * N * (N - 1) / 2
    term_2b = (3 + 2 * np.sqrt(2)) * d ** 2 - (2 - np.sqrt(2)) * d ** (5 / 2) - (7 / 2 + 3 * np.sqrt(2)) * d ** 3 + \
              (1 + np.sqrt(2)) * d ** (7 / 2) + (3 / 2 + np.sqrt(2)) * d ** 4
    return (1 / epsilon ** 2) * (term_1 + term_2a * term_2b) ** 2


def shots_grouped_zero_state(N, d, a, r):
    gam = gamma(a, r)
    epsilon = delta_E(gam)
    if N % 2 == 0:
        N_term = abs(gam) * np.sqrt(N / 2) * (N - 1)
    else:
        N_term = abs(gam) * np.sqrt((N - 1) / 2) * N
    return (1 / epsilon ** 2) * (1 / 16) * N_term ** 2 * d ** 4


def shots_ungrouped_zero_state(N, d, a, r):
    gam = gamma(a, r)
    epsilon = delta_E(gam)
    N_term = abs(gam) * N * (N - 1) / 2
    return (1 / epsilon ** 2) * (1 / 16) * N_term ** 2 * d ** 4


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                        **orig_handle[1], label='a')
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                        **orig_handle[0], label='b')
        return [l1, l2]


tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 9*1.5,
    "font.size": 9*1.5,
    "legend.fontsize": 9*1.5,
    "xtick.labelsize": 9*1.5,
    "ytick.labelsize": 9*1.5,
    "axes.titlesize": 9*1.5,
    "savefig.facecolor": (0, 0, 0, 0)
}
plt.rcParams.update(tex_fonts)

molecules = {'I2': (r'I\textsubscript{2}', 10.4, 4),
             'CS2': (r'CS\textsubscript{2}', 8.8, 3.7),
             # 'CCl4': (r'CCl\textsubscript{4}', 10, 4),
             'H2O': (r'H\textsubscript{2}O', 1.6, 2.7),
             # 'PBr3': (r'PBr\textsubscript{3}', 10.4, 3.5)
             }

# molecules = {'I2': (r'I\textsubscript{2}', 10.4, 4)}

fig, (ax_1, ax_3) = plt.subplots(1, 2, figsize=(6.75, 2.5))
fig.subplots_adjust(hspace=0.5)

fixed_N = 2  # fixed N
fixed_d = 4  # fixed d

varied_Ns = range(2, 11)
varied_ds = 2 ** np.array(range(1, 5))

linestyles1 = mpltex.linestyle_generator(markers=[], lines=['-'])
linestyles2 = mpltex.linestyle_generator(markers=[], lines=['--'])

handles, labels = [], []
lines = []

for (label, a, r) in molecules.values():
    linestyle1 = next(linestyles1)
    linestyle2 = next(linestyles2)

    shots_grouped_uniform_measure_fixed_d = [shots_grouped_uniform_measure(N, fixed_d, a, r) for N in varied_Ns]
    shots_ungrouped_uniform_measure_fixed_d = [shots_ungrouped_uniform_measure(N, fixed_d, a, r) for N in varied_Ns]
    shots_grouped_zero_state_fixed_d = [shots_grouped_zero_state(N, fixed_d, a, r) for N in varied_Ns]
    shots_ungrouped_zero_state_fixed_d = [shots_ungrouped_zero_state(N, fixed_d, a, r) for N in varied_Ns]

    shots_grouped_uniform_measure_fixed_N = [shots_grouped_uniform_measure(fixed_N, d, a, r) for d in varied_ds]
    shots_ungrouped_uniform_measure_fixed_N = [shots_ungrouped_uniform_measure(fixed_N, d, a, r) for d in varied_ds]
    shots_grouped_zero_state_fixed_N = [shots_grouped_zero_state(fixed_N, d, a, r) for d in varied_ds]
    shots_ungrouped_zero_state_fixed_N = [shots_ungrouped_zero_state(fixed_N, d, a, r) for d in varied_ds]

    ax_1.plot(varied_Ns, shots_grouped_uniform_measure_fixed_d, **linestyle1, linewidth=1*1.5, label=label + ' (grouped)')
    ax_1.plot(varied_Ns, shots_ungrouped_uniform_measure_fixed_d, **linestyle2, linewidth=1*1.5,
              label=label + ' (ungrouped)')

    ax_3.plot(varied_ds, shots_grouped_uniform_measure_fixed_N, **linestyle1, linewidth=1*1.5, label=label + ' (grouped)')
    ax_3.plot(varied_ds, shots_ungrouped_uniform_measure_fixed_N, **linestyle2, linewidth=1*1.5,
              label=label + ' (ungrouped)')

for ax in [ax_1, ax_3, ]:
    ax.set_yscale('log')
    ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True,width=0.75*1.5)
    for side in ["bottom","top", "left","right"]:
        ax.spines[side].set_linewidth(0.75*1.5)

for ax, subplot in zip([ax_1, ], ['a', ]):
    ax.set_xticks(varied_Ns)
    ax.set_xlabel(r'$N$')

    label = r'\textbf{(' + subplot + ')} $d=$' + f' {fixed_d}'
    ax.text(s=label, x=0.05, y=0.85, transform=ax.transAxes)
    ax.set_ylabel('Shots')

for ax, subplot in zip([ax_3, ], ['b', ]):
    ax.set_xticks(varied_ds)
    ax.set_xlabel(r'$d$')

    label = r'\textbf{(' + subplot + ')} $N=$' + f' {fixed_N}'
    ax.text(s=label, x=0.05, y=0.85, transform=ax.transAxes)
# fig.tight_layout()

fig.subplots_adjust(top=0.95,bottom=0.45,left=0.1,right=0.9)
legend = ax_3.legend(fancybox=False, edgecolor="k", framealpha=1, ncol=len(molecules),
                     loc="lower center", bbox_to_anchor=(0.5, -0.02), bbox_transform=fig.transFigure)
legend.get_frame().set_linewidth(.75*1.5)
fig.savefig('shots_scaling.pdf')  # , bbox_inches='tight')
fig.show()
