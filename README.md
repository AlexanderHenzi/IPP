# Invariant Probabilistic Prediction
This repository contains implementations and replication material for the 
preprint

Alexander Henzi, Xinwei Shen, Michael Law, and Peter Bühlmann. "Invariant Probabilistic Prediction" arXiv preprint [arXiv:2309.10083](https://arxiv.org/abs/arXiv:2309.10083) (2023).

The folder ``illustrative_example`` contains code to
replicate the example from Section 3.3 of the preprint.
R and Python implementations of IPP and other methods applied in the case study are in ``functions``. Code for the simulation study in Section 5.1 of the preprint is in ``simulation_study``. The file ``simulation_study.R`` generates a single run of the simulations and can be submitted to a HPC cluster with SLURM. The collected simulation results are available in ``data``. Code for the single cell data application in Section 5.2 is in ``single_cell_application``. The selected test environments, as well as some intermediate results, are stored in ``data`` for convenience.
