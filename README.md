# Invariant Probabilistic Prediction

This repository contains implementations and replication material for the 
preprint

Alexander Henzi, Xinwei Shen, Michael Law, and Peter Bühlmann. "Invariant Probabilistic Prediction" arXiv preprint [arXiv:2309.10083](https://arxiv.org/abs/arXiv:2309.10083) (2023).

## General implementation of our methods

The folder `functions` contains implementations for our methods in R and Python: `ipp.R` is the R implementation for the parametric IPP, and `ipp_nn.py` is the python implementation of the neural network variant. For the package requirements in Python, see the `requirements.txt` file in the main directory.

Other methods used in the paper are in `distributional_anchor_regression.R` (for distributional anchor regression, see https://github.com/LucasKook/distributional-anchor-regression), `anchor_drig.py` (for anchor regression and DRIG, see https://github.com/xwshen51/DRIG/blob/main/estimate.py), and `cputils.py` (for conformal prediction, see https://github.com/zhimeir/finegrained-conformal-paper). The conformal prediciton methdos rely on the `qosa` package in Python (https://gitlab.com/qosa_index/qosa), which is included in the `functions` directory for easy reference.

## Reproducing the toy example

Code for the illustrative example from Section 2.4 of the article is in the directory `illustration`.

## Reproducing the simulations

The folder `simulation_study` contains the code to reproduce the simulations in the article and supplement/appendix: `simulation_study.R` and `simulation_study_results.R` are for the simulation study in the article.

For the simulations in the article, `simulation_study.R` computes a single run of the simulations for a certain parameter combination, which has to be parallelized on a HPC cluster to reproduce the complete simulations (with `array_task_simulation.sh`); `simulation_study_results.R` collects the results and generates the figures. The collected results can be found in `data/simulation_results_logs/scrps.rda`, so that the figures are reproducible without running all simulations.

For the simulations in the supplement/appendix, `simulation_study_2_data_preparation.R` generates the semi-real single cell data for all simulation runs, and `simulation_study_2_computations.R` computes the results on a single run (to be parallelized with `array_task_simulation_2.R`). The results are collected and processed with `simulation_study_2_results.R`. The simulation results are contained in `data/simulation_2_results`.

## Reproducing the single cell application

Scripts for the single cell data application are in `single_cell_application`. The script generating all the figures in the article is `single_cell_application.R`. Since the application involves many different methods, some of which are computationally expensive and some of which are implemented in R and others in Python, the computations are done in different scripts. The parametric variants of IPP and distributional anchor regression are done in `single_cell_application.R`. The computations are slow and intermediate results are stored in `data` and automatically loaded when the `run_computations` parameter is set to `FALSE`. Anchor regression and DRIG are computed with `single_cell_anchor_drig.py`, the neural network variants of IPP with `single_cell_ipp_nn.py`, and V-REx with `single_cell_vrex.py`. The conformal prediction methods have long computation time, and `single_cell_cp.py` applies them for a single environment and penalty parameter (parallelized on a cluster with `array_task_single_cell_cp.py`).
