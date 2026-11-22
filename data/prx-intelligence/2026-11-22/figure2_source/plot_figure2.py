"""Reproduce the single-column optimizer figure from the two supplied CSV files.

Run with Python, numpy, pandas, matplotlib and PyMuPDF installed. The original
per-step energies are validated against every supplied nonoverlapping block.
Total energies are converted to meV and divided by N=12 electrons.
No smoothing, clipping, or seed averaging is applied.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pymupdf

HERE = Path(__file__).resolve().parent
RAW_NAME = 'figure_s1_energy_vs_step_with_complex_minsr.csv'
BLOCK_NAME = 'figure_s1_plot_data.csv'
RAW_TO_DISPLAY = {
    'complex_minsr': 'MinSR (Complex)',
    'complex_amplitude_kfac': 'KFAC (Complex)',
    'real_minsr': 'MinSR (Real)',
    'real_kfac': 'KFAC (Real)',
}
raw = pd.read_csv(HERE / RAW_NAME)
blocks = pd.read_csv(HERE / BLOCK_NAME)
assert raw.shape == (80000, 4) and blocks.shape == (800, 7)
assert not raw.isna().any().any() and not blocks.isna().any().any()
assert set(raw.series) == set(RAW_TO_DISPLAY)
assert not raw.duplicated(['n_phi', 'series', 'step']).any()
assert np.isfinite(raw.energy_optimized).all()
for _, group in raw.groupby(['n_phi', 'series']):
    assert np.array_equal(np.sort(group.step), np.arange(1, 10001))
raw['series'] = raw.series.map(RAW_TO_DISPLAY)
raw['block'] = (raw.step - 1) // 100
recomputed = raw.groupby(['n_phi', 'series', 'block']).agg(
    step_start=('step', 'min'), step_end=('step', 'max'),
    energy_mean=('energy_optimized', 'mean'),
    energy_min=('energy_optimized', 'min'),
    energy_max=('energy_optimized', 'max')).reset_index()
checked = recomputed.merge(blocks, on=['n_phi', 'series', 'step_start', 'step_end'],
                           validate='one_to_one', suffixes=('_raw', '_supplied'))
assert len(checked) == 800
differences = {key: float(np.max(np.abs(checked[key+'_raw']-checked[key+'_supplied'])))
               for key in ('energy_mean', 'energy_min', 'energy_max')}
assert max(differences.values()) < 1e-12

# Use the paper's effective atomic units (m*/m_e=0.35, epsilon_r=5).
# Convert total energy to energy per electron, consistent with Table I.
HARTREE_MEV, M_STAR, EPSILON_R = 27211.386245988, .35, 5.0
HA_STAR_MEV = HARTREE_MEV * M_STAR / EPSILON_R**2
N_ELECTRONS = 12
ENERGY_SCALE = HA_STAR_MEV / N_ELECTRONS
display = blocks.copy()
for key in ('energy_mean', 'energy_min', 'energy_max'):
    display[key] = blocks[key] * ENERGY_SCALE
    assert np.allclose(display[key]/ENERGY_SCALE, blocks[key], atol=1e-14, rtol=0)
display.rename(columns={k:k+'_meV_per_electron' for k in ('energy_mean','energy_min','energy_max')}).to_csv(
    HERE/'figure2_plot_data_meV_per_electron.csv', index=False)

# Retain the 88 mm canvas and blue/ochre palette; strengthen axes to match
# Figure 3 and use two separated frames and one compact, bordered legend.
WIDTH_PT, HEIGHT_PT = 249.44882, 184.0
BLUE, OCHRE = '#376F96', '#A45231'
INK = '#222222'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7.1, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4,
    'mathtext.fontset': 'dejavusans', 'text.color': INK,
    'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': .7, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.pad': 1.6, 'ytick.major.pad': 1.6,
    'axes.spines.top': True, 'axes.spines.right': True,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
fig = plt.figure(figsize=(WIDTH_PT/72, HEIGHT_PT/72))
axes = [fig.add_axes([.151, .172, .348, .576]),
        fig.add_axes([.629, .172, .348, .576])]
styles = {
    'MinSR (Complex)': (BLUE, '-', 'o'),
    'KFAC (Complex)': (BLUE, (0, (3.5, 2.0)), 'o'),
    'MinSR (Real)': (OCHRE, '-', 'D'),
    'KFAC (Real)': (OCHRE, (0, (3.5, 2.0)), 'D'),
}
for ax, flux, letter in zip(axes, (1, 2), 'ab'):
    # The supplied block end is the x-coordinate; use 10^3 only as an axis unit.
    for label, (color, dash, marker) in styles.items():
        data = display[(display.n_phi == flux) & (display.series == label)].sort_values('step_end')
        x = data.step_end.to_numpy()/1000
        ax.fill_between(x, data.energy_min, data.energy_max, color=color,
                        alpha=.14, linewidth=0, zorder=1)
    for label, (color, dash, marker) in styles.items():
        data = display[(display.n_phi == flux) & (display.series == label)].sort_values('step_end')
        ax.plot(data.step_end.to_numpy()/1000, data.energy_mean,
                color=color, linestyle=dash, linewidth=.95,
                marker=marker, markevery=[19, 39, 59, 79, 99], markersize=2.25,
                markerfacecolor='white', markeredgewidth=.55, zorder=3,
                clip_on=False)
    ax.set(xlim=(0, 10.2), ylim=(-62.5, 0), xticks=[0, 5, 10],
           yticks=[-60, -40, -20, 0])
    ax.set_title(r'$n_\phi = '+str(flux)+'$', pad=4.7)
    ax.text(-.03, 1.045, letter, transform=ax.transAxes,
            fontsize=8.7, fontweight='bold', ha='left', va='bottom')
axes[0].set_ylabel(r'$E/N$ (meV/electron)', labelpad=2)
fig.text(.5535, .038, r'Optimization step ($\times 10^3$)', ha='center', va='bottom', fontsize=7.1)
handles = [Line2D([], [], color=color, linestyle=dash, linewidth=.95,
                  marker=marker, markersize=2.4, markerfacecolor='white',
                  markeredgewidth=.55, label=label)
           for label, (color, dash, marker) in styles.items()]
legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .996),
           ncol=2, frameon=True, fancybox=False, shadow=False,
           facecolor='white', edgecolor=INK, framealpha=1,
           fontsize=6.5, handlelength=2.15, borderpad=.55,
           handletextpad=.5, columnspacing=1.35, labelspacing=.45, borderaxespad=.35)
legend.get_frame().set_linewidth(.6)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
figure_box = fig.get_window_extent(renderer)
text_boxes = [(artist.get_text(), artist.get_window_extent(renderer))
              for artist in fig.findobj(matplotlib.text.Text)
              if artist.get_visible() and artist.get_text()]
outside = [text for text, box in text_boxes
           if box.x0 < figure_box.x0-.5 or box.y0 < figure_box.y0-.5
           or box.x1 > figure_box.x1+.5 or box.y1 > figure_box.y1+.5]
assert not outside, outside
output = HERE.parent / 'figs/figure7_optimizer_comparison.pdf'
output.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(output, metadata={'Title': 'Figure 2: optimizer comparison',
                             'Author': '', 'CreationDate': None, 'ModDate': None})
fig.savefig(HERE / 'figure2_single_column.svg')
svg_path = HERE / 'figure2_single_column.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
plt.close(fig)
with pymupdf.open(output) as pdf:
    page = pdf[0]
    assert abs(page.rect.width-WIDTH_PT) < .01
    page.get_pixmap(matrix=pymupdf.Matrix(3, 3)).save(HERE / 'figure2_preview.png')
    page.get_pixmap(matrix=pymupdf.Matrix(3, 3), colorspace=pymupdf.csGRAY).save(HERE / 'figure2_grayscale.png')
record = {
    'figure': 'Figure 2', 'raw_rows': len(raw), 'block_rows': len(blocks),
    'series_mapping': RAW_TO_DISPLAY, 'flux_values': [1, 2],
    'updates_per_trajectory': 10000, 'block_size': 100,
    'summary': 'Arithmetic mean in nonoverlapping 100-update blocks, plotted at block end.',
    'band': 'Within-block minimum and maximum; not a confidence interval or seed variability.',
    'kfac_complex_scope': 'Amplitude-only KFAC, as named in the supplied raw CSV and current manuscript.',
    'data_recomputation_max_abs_error': differences,
    'width_pt': WIDTH_PT, 'height_pt': HEIGHT_PT,
    'energy_unit': 'meV/electron', 'energy_quantity': 'training energy per electron',
    'n_electrons': N_ELECTRONS,
    'input_energy_unit': 'effective Hartree',
    'conversion': {'hartree_meV': HARTREE_MEV, 'm_star_over_m_e': M_STAR,
                   'epsilon_r': EPSILON_R, 'Ha_star_meV': HA_STAR_MEV,
                   'formula': 'Ha_star_meV = hartree_meV * m_star_over_m_e / epsilon_r**2',
                   'per_electron_scale': ENERGY_SCALE,
                   'display_formula': 'energy_meV_per_electron = total_energy_Ha_star * Ha_star_meV / n_electrons'},
    'conversion_source': 'Manuscript Sec. II and Appendix A; project systems/units.py (CODATA 2018 constant).',
    'axes_color': INK, 'axes_linewidth_pt': .7, 'legend_border': True,
    'panel_gap_pt': (.629-.151-.348)*WIDTH_PT,
    'palette': {'Complex': BLUE, 'Real': OCHRE},
    'line_styles': {'MinSR': 'solid', 'KFAC': 'dashed'},
    'markers': {'Complex': 'open circle', 'Real': 'open diamond'},
    'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                      for name in (RAW_NAME, BLOCK_NAME)},
    'pdf_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'text_within_canvas': True,
}
(HERE / 'provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
