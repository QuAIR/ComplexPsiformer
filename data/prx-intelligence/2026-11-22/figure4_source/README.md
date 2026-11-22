# Figure 4 source and display processing

Run `python plot_figure4.py` with NumPy, Matplotlib and PyMuPDF to regenerate
`../figs/figure4_continuum.pdf` and a preview in this folder.

Panel (a) uses the same 229,376 retained configurations as the original figure,
from the standard-start 20,000-update state at N=12, nu=1/3 and kappa=9. Each
configuration's pair displacements were rotated by 0, 60, 120, 180, 240 and
300 degrees in the hexagonal torus before histogramming, then averaged with
equal weights. The rotations are not independent samples.

For display, the periodically extended C6-averaged field is rotated
counterclockwise by 11.704391616748474 degrees inside a fixed hexagonal
display window. The two panels' hexagonal outlines retain the same orientation.
The mesh is extended using the rotated lattice to cover the entire fixed crop;
the outline is a display window, not the boundary of the rotated periodic cell.
This aligns the phase of the
sixth angular harmonic in 0.85 < r/a < 1.15 with that of panel (b). The bin
values and sixfold amplitude are unchanged by this display rotation. It is
not a symmetry projection at the original fixed lattice orientation.

Panel (b) retains the original nu=3/10 histogram and orientation. The shared
color scale remains 0--2.5, with the finite-cell mean at 11/12. No additional
spatial smoothing is applied.

Panel (c), its uncertainty bands and the shared colorbar are preserved from
the previous figure in `figure4_lower_panels.pdf`. Panel (c) was evaluated
from the original, unaveraged samples in the unrotated coordinate frames;
its numbers were not recomputed from the orientation-aligned display map.

`figure4_data.npz` retains both original histograms, the original angular
contrasts and standard errors, and adds `nu_one_third_c6` and
`display_rotation_degrees`. `provenance.json` records the frozen-model and
sample hashes, transformation scope and prior figure hash. The plot source
uses `figure4_lower_panels.pdf` to preserve the original lower-panel artwork.
