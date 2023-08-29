import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

output_file = 'grouping_and_truncation_scaling_gamma_0.5.npy'

output = np.load(output_file)

num_oscillators_values = [int(n) for n in list(set(output[:, 0]))]
qubits_per_oscillator_values = [int(m) for m in list(set(output[:, 1]))]

num_groups_results = np.empty((len(qubits_per_oscillator_values), len(num_oscillators_values)))
num_terms_results = np.full_like(num_groups_results, np.nan)
upper_bound_groups = np.full_like(num_groups_results, np.nan)
expected_num_terms = np.full_like(num_groups_results, np.nan)
exact_energies_results = np.full_like(num_groups_results, np.nan)
truncated_energies_results = np.full_like(num_groups_results, np.nan)
energies_fractional_error_results = np.full_like(num_groups_results, np.nan)

for (num_oscillators, qubits_per_oscillator, num_terms, num_groups, simp_ground_state_energy, trunc_ground_state_energy) \
        in output:

    num_groups_results[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = num_groups

    if num_oscillators == 1:
        upper_bound_groups[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = 1

    elif num_oscillators % 2 == 0:
        upper_bound_groups[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] \
            = int(1 + (2 ** qubits_per_oscillator - 1) ** 2 * (num_oscillators - 1))
    else:
        upper_bound_groups[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] \
            = int(1 + (2 ** qubits_per_oscillator - 1) ** 2 * (num_oscillators))

    num_terms_results[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = num_terms

    expected_num_terms[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] \
        = int(1 + qubits_per_oscillator * num_oscillators
              + (1 / 2) * (2 ** qubits_per_oscillator * qubits_per_oscillator / 2) ** 2
              * (num_oscillators - 1) * num_oscillators)

    exact_energies_results[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = simp_ground_state_energy
    truncated_energies_results[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = trunc_ground_state_energy

    if num_oscillators == 1:
        energy_error = np.nan
    else:
        energy_error \
            = abs((simp_ground_state_energy - trunc_ground_state_energy) / (num_oscillators - simp_ground_state_energy))
    energies_fractional_error_results[int(qubits_per_oscillator) - 1, int(num_oscillators) - 1] = energy_error

x, y = np.meshgrid(num_oscillators_values, [2 ** q for q in qubits_per_oscillator_values])

fig0 = plt.figure()
ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
ax0.plot_wireframe(x, y, num_groups_results)
ax0.set_xticks(num_oscillators_values)
ax0.set_yticks([2 ** q for q in qubits_per_oscillator_values])
ax0.set_xlabel('Oscillators')
ax0.set_ylabel('Fock states')
ax0.set_zlabel('Circuits')

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_xticks(num_oscillators_values)
ax1.set_yticks(qubits_per_oscillator_values)
cs = ax1.imshow(num_groups_results, norm=LogNorm(),
                extent=[min(num_oscillators_values) - 0.5, max(num_oscillators_values) + 0.5,
                        min(qubits_per_oscillator_values) - 0.5, max(qubits_per_oscillator_values) + 0.5])
cbar = fig1.colorbar(cs)
cbar.set_label('Circuits')
ax1.set_xlabel('Oscillators')
ax1.set_ylabel('Qubits per oscillator')

fig2, ((ax2_1, ax2_3), (ax2_2, ax2_4)) = plt.subplots(2, 2, figsize=(6, 2.5))
fig2.subplots_adjust(hspace=0.1)

m1 = 2
m2 = 4

# ax2_1.plot(num_oscillators_values, num_terms_results[m1 - 1, :], 'b:')
ax2_1.plot(num_oscillators_values, num_groups_results[m1 - 1, :], 'k', linewidth=1)
ax2_1.plot(num_oscillators_values, upper_bound_groups[m1 - 1, :], 'r--', linewidth=1)
# ax2_1.set_xlabel('$N$')
ax2_1.set_ylabel('Circuits')
ax2_1.set_xticks(num_oscillators_values)
ax2_1.text(s=r'\textbf{(a)} $d=$' + f' {2 ** m1}', x=0.05, y=0.775, transform=ax2_1.transAxes)
ax2_1.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True, labelbottom=False)

# ax2_2.plot(num_oscillators_values, num_terms_results[m2 - 1, :], 'b:')
ax2_2.plot(num_oscillators_values, num_groups_results[m2 - 1, :], 'k', linewidth=1)
ax2_2.plot(num_oscillators_values, upper_bound_groups[m2 - 1, :], 'r--', linewidth=1)
ax2_2.set_xlabel('$N$')
ax2_2.set_ylabel('Circuits')
ax2_2.set_xticks(num_oscillators_values)
ax2_2.text(s=r'\textbf{(b)} $d=$' + f' {2 ** m2}', x=0.05, y=0.775, transform=ax2_2.transAxes)
ax2_2.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

N1 = 2
N2 = 5

ax2_3.plot([2 ** m for m in qubits_per_oscillator_values], num_groups_results[:, N1 - 1], 'k', linewidth=1)
ax2_3.plot([2 ** m for m in qubits_per_oscillator_values], upper_bound_groups[:, N1 - 1], 'r--', linewidth=1)
# ax2_3.set_xlabel('$d$')
ax2_3.set_xticks([2 ** m for m in qubits_per_oscillator_values])
ax2_3.text(s=r'\textbf{(c)} $N=$' + f' {N1}', x=0.05, y=0.775, transform=ax2_3.transAxes)
ax2_3.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True, labelbottom=False)

ax2_4.plot([2 ** m for m in qubits_per_oscillator_values], num_groups_results[:, N2 - 1], 'k', linewidth=1)
ax2_4.plot([2 ** m for m in qubits_per_oscillator_values], upper_bound_groups[:, N2 - 1], 'r--', linewidth=1)
ax2_4.set_xlabel('$d$')
ax2_4.set_xticks([2 ** m for m in qubits_per_oscillator_values])
ax2_4.text(s=r'\textbf{(d)} $N=$' + f' {N2}', x=0.05, y=0.775, transform=ax2_4.transAxes)
ax2_4.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

for ax in [ax2_1, ax2_2, ax2_3, ax2_4]:
    ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

fig2.tight_layout()
fig2.savefig('circuits_scaling.pdf')
fig2.show()
