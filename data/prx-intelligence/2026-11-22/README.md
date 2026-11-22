# PRX Intelligence data snapshot: 22 November 2026

Numerical data, figure sources, diagnostics, and artwork transferred from the
`PRX Intelligence` Overleaf submission directory at source revision
`b46e4e841e3d1e75db39a817bb929926836de365`.
The 25 source files are preserved byte for byte. `manifest.json` records their
sizes and SHA-256 hashes. The manifest and this README are added archive metadata.

| Directory | Contents |
|---|---|
| `benchmark-data/` | All 27 endpoint energies for Table I, conditional sampling errors, seedwise results, and diagnostic provenance. |
| `figure2_source/` | Eight optimizer trajectories, 100-update block summaries, per-electron plotting data, plotting script, previews, and provenance. |
| `figure4_source/` | Original and C6-averaged correlation data, display-rotation provenance, plotting script, and preserved lower-panel artwork. |
| `figs/` | Final manuscript figure PDFs and other artwork preserved from the source directory. |

The four PDFs used by the manuscript are:

- Figure 1: `figs/figure1_method.pdf`
- Figure 2: `figs/figure7_optimizer_comparison.pdf`
- Figure 3: `figs/filling_charge_order_single_column.pdf`
- Figure 4: `figs/figure4_continuum.pdf`

The additional SVG and PDF files in `figs/` were not referenced by that revision
of the manuscript and are retained here as source/history material.

The original directory structure is retained so that the plotting scripts can
write to their expected `../figs/` paths. To regenerate a figure, use a separate
working copy so the archived files and their hashes remain available:

```sh
python figure2_source/plot_figure2.py
python figure4_source/plot_figure4.py
```

Dependencies and processing details are documented in each source directory.
Table I reports variation across three training seeds; its per-state conditional
sampling errors and flags remain in the data files. Figure 2 bands are within-block
minimum-to-maximum ranges. Figure 4's angular contrast comes from the original,
unaveraged samples in their unrotated frames; the panel (a) display transformation
does not change those recorded values.

Earlier numerical inputs for Figure 3 and the supplementary comparisons remain in
the adjacent `../supplemental-data.zip`. That legacy archive also contains
historical experiments; the two-electron CI study is not used in the current
manuscript. This snapshot does not add raw Monte Carlo configuration streams or
trained checkpoints.
