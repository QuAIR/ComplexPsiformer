"""Reproduce the Table S3 optimization curves from the accompanying CSV."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = np.genfromtxt(HERE/'training_energy.csv', delimiter=',', names=True)
assert np.array_equal(data['update'], np.arange(1, 20001))
assert np.isfinite(data.tolist()).all()
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'axes.linewidth': 0.65,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'lines.linewidth': 1.3, 'savefig.dpi': 240,
})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), layout='constrained')
block = 100
x = data['update'].reshape(-1, block).mean(axis=1)/1000
curves = [
    ('negligible_energy_meV_per_electron', 'Negligible angular contrast', '#0072B2', '-'),
    ('six_peaks_energy_meV_per_electron', 'Six first-shell peaks', '#D55E00', '-'),
]
for field, label, color, style in curves:
    y = data[field].reshape(-1, block).mean(axis=1)
    for ax in axes:
        ax.plot(x, y, color=color, linestyle=style, label=label)
for i, ax in enumerate(axes):
    ax.set_xlabel(r'MinSR updates ($10^3$)')
    ax.set_ylabel('Energy (meV/electron)')
    ax.tick_params(direction='in', top=True, right=True, width=.65, length=3)
    ax.grid(axis='y', color='#E5E5E5', linewidth=.4)
    ax.set_axisbelow(True)
    ax.set_title('(a) Full training' if i == 0 else '(b) Late training', loc='left', pad=5)
axes[0].set_xlim(0, 20)
axes[0].set_xticks([0, 5, 10, 15, 20])
axes[0].set_ylim(-14.5, -8)
axes[0].set_yticks([-14, -12, -10, -8])
axes[0].legend(loc='upper right', frameon=False, handlelength=2.7)
axes[1].set_xlim(15, 20)
axes[1].set_xticks([15, 16, 17, 18, 19, 20])
axes[1].set_ylim(-14.21, -13.90)
axes[1].set_yticks([-14.2, -14.1, -14.0, -13.9])
for suffix in ['pdf', 'png']:
    fig.savefig(HERE/f'uniform_training_states.{suffix}', bbox_inches='tight')
plt.close(fig)

