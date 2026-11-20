# Data source of the Severson segments

**Original dataset:** K.A. Severson, P.M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M.H. Chen, M. Aykol, P.K. Herring, D. Fraggedakis, M.Z. Bazant, S.J. Harris, W.C. Chueh, R.D. Braatz. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019).

**Source:** Toyota Research Institute data portal, https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 (three batch files, 124 LFP/graphite cells, nominal capacity 1.1 Ah).

**Licence:** the original data are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ .

**Derived data in this folder.** The files in `samples/`, `metadata.csv` and `dataset_meta.json` are derived from the original data. Changes made:

- the 124 cells are assembled from the batch files following the original authors' convention (records of cells that did not reach end of life are dropped; continuation records are appended to their cells);
- each cycle is split into a charge segment (the fast-charge part C1 → Q1 → C2, up to the rest before the common 1C constant-current/constant-voltage top-off) and a discharge segment (the full discharge);
- every tenth cycle is kept, plus the first and last usable cycle of each cell; segments failing quality checks (missing or non-monotonic data, logging gaps or pauses, implausible length or capacity) are excluded;
- each segment is linearly resampled to 512 points and stored with two channels: voltage (V) and local capacity (Ah, counted from the start of the segment), as `x_raw` in physical units and `x_norm` scaled to [0, 1] per segment;
- each segment carries its cycle number, step type and charging-policy set-points (C1, Q1, C2), with the set-points of five cells corrected to the values actually applied (listed in `dataset_meta.json`), and the original train / primary test / secondary test split, with seven training cells held out for validation.

These derived data remain available under CC BY 4.0. If you use them, please cite the original article above.
