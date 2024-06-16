#!/bin/bash
#SBATCH --mail-type=end,fail
#SBATCH --job-name="ipp_single_Cell_cp"
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-899

#### Your shell commands below this line ####

## load modules: replace ??? by latest version
module load python/???
python3 single_cell_cp.py