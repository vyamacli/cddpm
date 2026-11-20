# Data source of the NASA segments

**Original dataset:** B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

**Source:** NASA Prognostics Center of Excellence data set repository, https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ ; NASA Open Data Portal entry, https://data.nasa.gov/dataset/li-ion-battery-aging-datasets (listed as public, licence not specified).

**Terms stated by the source:** "Publications making use of databases obtained from this repository are requested to acknowledge both the assistance received by using this repository and the donators of the data." and "Users employ the data at their own risk. Neither NASA nor the donators of the data sets assume any liability for the use of the data, or any system developed using the data."

**Acknowledgement:** we acknowledge the NASA Prognostics Data Repository (NASA Ames Research Center) and the donors of the data, B. Saha and K. Goebel.

**Derived data in this folder.** The files in `samples/`, `metadata.csv` and `dataset_meta.json` are derived from the original data:

- 33 cells (B0005 to B0056, B0052 excluded) are read from the .mat files of the data set; exact duplicate files are removed;
- every charge and discharge operation is taken as a segment; non-finite and duplicate time points are removed, and segments that are too short (fewer than 40 points or under 20 s), have implausible voltage or near-zero current, or (discharge) lack a positive recorded capacity are excluded;
- local capacity (Ah) is computed by cumulative trapezoidal integration of the absolute measured current over time; on discharge segments it is scaled so that its end value equals the capacity recorded in the data set for that cycle;
- each segment is linearly resampled to 512 points and stored with two channels: voltage (V) and local capacity (Ah), as `x_raw` in physical units and `x_norm` scaled to [0, 1] per segment, together with a condition vector `cond` (cell index, cycle position, mean current and step type, each scaled to [0, 1]);
- `metadata.csv` lists each segment with its cell, cycle number, step, split, ambient temperature, current and voltage summaries and the source file name and checksum.

The original data remain the property of their provider. This repository grants no licence for the original data; anyone reusing them should obtain them from the source above and follow its terms.
