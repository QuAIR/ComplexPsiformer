# Figure 2: single-column optimizer comparison

Run `python plot_figure2.py` in this directory (or by absolute path). Requires
NumPy, pandas, Matplotlib, and PyMuPDF. Arial is used when installed, with
DejaVu Sans as fallback. Output: `../figs/figure7_optimizer_comparison.pdf`.

Both CSV files are unchanged copies supplied by the author on 2026-11-21.
The script checks every mean/minimum/maximum against the raw 10,000-step
trajectory before plotting. These are total training energies in Ha*;
the shaded bands are within-block ranges, not confidence intervals.
The input CSV energies are in effective Hartrees. The display and
`figure2_plot_data_meV_per_electron.csv` use energy per electron in
meV/electron, with the manuscript's
effective mass m*/m_e=0.35 and relative permittivity epsilon_r=5:
1 Ha* = 27211.386245988 * 0.35 / 5^2 = 380.95940744383194 meV.
All trajectories contain N=12 electrons. Every block mean, minimum, and
maximum is multiplied by 380.95940744383194 / 12 to obtain meV/electron,
matching the energy normalization and units of Table I. These curves retain
the training-energy estimator; the Table I values are not changed.
The original input CSV files are preserved byte for byte.
The Complex KFAC series is the amplitude-only implementation identified in
the input CSV and manuscript. No claim about other KFAC implementations is added.

The 88 mm canvas matches Figures 3 and 4. Blue (#376F96) denotes Complex,
ochre (#A45231) denotes Real; solid/dashed curves distinguish MinSR/KFAC.
Axes and ticks use near-black (#222222), with 0.7 pt rectangular frames.
The two panels have independent tick labels and a 32.43 pt horizontal gap.
One thin rectangular frame encloses the shared two-column legend.
Sparse open circles/diamonds provide a redundant model label in grayscale.
Provenance, hashes, and numerical validation are recorded in `provenance.json`.

The manuscript groups Table I and Figure 2 in one single-column float, with
separate numbered captions and labels, so the table and figure stay together.
The optimizer-comparison paragraphs are in Results immediately after this
display; the optimizer definitions remain in the method section.
