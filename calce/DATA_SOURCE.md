# Data source of the CALCE segments

**Original dataset:** CALCE battery data, CALCE Battery Research Group, University of Maryland, https://calce.umd.edu/battery-data (CX2 cells; cells CX2_33 to CX2_38 are used here).

**Terms stated by the source:** the page states "We provide open access to our experimental test data on lithium-ion batteries" and "Use of this data for publication purposes should include references to the CALCE article(s) that describe the experiments conducted to generating the data (Suggested citation)." The page names no licence.

**References cited for this dataset in the associated article:**

- CALCE Battery Research Group, CALCE battery data, https://calce.umd.edu/battery-data.
- Y. Xing, E.W.M. Ma, K.-L. Tsui, M. Pecht, An ensemble model for predicting the remaining useful performance of lithium-ion batteries, Microelectronics Reliability 53 (6) (2013) 811–820.

**Derived data in this folder.** The files in `samples/`, `metadata.csv` and `dataset_meta.json` are derived from the original data:

- the cycling files of each cell are merged in date order and given a running cycle number;
- from each cycle the charge step (step index 2) and the discharge step (step index 7) are taken as segments; segments with fewer than 40 points are dropped;
- each segment is linearly resampled to 512 points and stored with two channels: voltage (V) and local capacity (Ah, the charge or discharge capacity counter relative to its value at the start of the step), as `x_raw` in physical units and `x_norm` scaled to [0, 1] per segment, together with a condition vector `cond` (cell index, cycle position, mean current and step type, each scaled to [0, 1]);
- `metadata.csv` lists each segment with its cell, cycle number, step, source file name and summary values.

The original data remain the property of their provider. This repository grants no licence for the original data; anyone reusing them should obtain them from the source above and follow its terms.
