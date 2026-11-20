# Conditional Diffusion Model for Probabilistic and Controllable Generation of Synthetic Battery Charge–Discharge Curves

**Authors:** Volkan Yamaçlı, Timur Lale, Serdar Ekinci, Rıdvan Fırat Çınar, Davut İzci, Gökhan Yüksek

## Project Description

This repository accompanies a study on conditional denoising diffusion probabilistic models (CDDPM) for generating synthetic lithium-ion battery charge and discharge curves. Charge and discharge steps from three public datasets (NASA PCoE, CALCE CX2 and Severson et al. 2019) are segmented, resampled to 512 points and represented as two synchronized channels: voltage and local capacity.

A one-dimensional conditional U-Net is trained as the denoising network. Given only the test-protocol settings of a step (such as step type, cycle position and the current or charging-policy settings of the dataset), the model generates synchronized voltage and local-capacity trajectories, and repeated sampling for the same settings yields a set of plausible curves rather than a single prediction. This repository contains the trained model and the processed segments of each dataset, together with scripts for inspecting the data and generating synthetic curves. The complete training and evaluation code will be released upon publication of the associated article.

## Usage

Run the scripts from the repository root (Python 3.10 or later with `numpy`, `pandas`, `torch`, `scipy`, `scikit-learn`, `matplotlib` and `seaborn`). The scripts are interactive when started without arguments; `--list` shows the available datasets and batteries and `--help` lists the options.

- `interactive_battery_signal_visualizer.py` plots real charge and discharge segments of selected batteries.
- `interactive_real_vs_synthetic_trajectory_generator.py` loads `best_model_ema.pt` of the selected dataset, generates synthetic trajectories for the protocol settings of a real segment and plots them against that segment, for example `python interactive_real_vs_synthetic_trajectory_generator.py --dataset nasa --battery B0006 --step discharge --count 5`. A GPU is used when available; on the CPU a run takes under a minute.
- `interactive_real_vs_synthetic_charge_discharge.py` shows, for a selected battery and life position, the real charge and discharge segments together with synthetic trajectories generated for the same protocol settings, in one figure.
- `interactive_reverse_diffusion_visualizer.py` shows the reverse diffusion process for the protocol settings of a real segment, from Gaussian noise to the final curves, as a row of panels and optionally as a GIF animation.
- `interactive_life_evolution_visualizer.py` draws the real charge or discharge segments of a selected battery from early to late life, colored by cycle number (real data only).

Each dataset folder (`calce/`, `nasa/`, `severson/`) holds the trained model (`best_model_ema.pt`), the processed segments (`samples/`), their metadata (`metadata.csv`, `dataset_meta.json`), the scaling constants used by the model (`global_stats.json`, `life_scale.json`) and a note on the data source (`DATA_SOURCE.md`).

## Datasets

The processed segments are derived from the following public datasets, which remain subject to their own terms of use; the source, the terms stated by the provider and the processing applied are described in `DATA_SOURCE.md` in each dataset folder.

- NASA PCoE: B. Saha, K. Goebel, Battery Data Set, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007. The repository and the donors of the data are gratefully acknowledged.
- CALCE CX2: CALCE Battery Research Group, University of Maryland, https://calce.umd.edu/battery-data; Y. Xing, E.W.M. Ma, K.-L. Tsui, M. Pecht, An ensemble model for predicting the remaining useful performance of lithium-ion batteries, Microelectronics Reliability 53 (6) (2013) 811–820.
- Severson et al. 2019: K.A. Severson et al., Data-driven prediction of battery cycle life before capacity degradation, Nature Energy 4 (2019) 383–391; https://data.matr.io/1/projects/5c48dd2bc625d700019f3204. The original data are released under CC BY 4.0, and the derived segments in `severson/` remain available under CC BY 4.0; the notice below does not restrict them.

No license for the original NASA and CALCE data is granted by this repository; obtain them from their sources and follow their terms.

## Copyright and Usage Restrictions

Copyright © 2026 Volkan Yamaçlı, Timur Lale, Serdar Ekinci, Rıdvan Fırat Çınar, Davut İzci, and Gökhan Yüksek.

All rights reserved.

No permission is granted to copy, modify, reproduce, distribute, publish, redistribute, sublicense, sell, incorporate into another project, or use the contents of this repository, in whole or in part, without prior written permission from the copyright holders.

The repository is provided solely for viewing, academic evaluation, and verification of the methodology described in the associated manuscript.

The public availability of this repository does not constitute an open-source license or permission for reuse.

Any third-party libraries, software components, and datasets referenced in this repository remain subject to their respective licenses and terms of use.

## Citation

The citation information for the associated article will be added after publication.
