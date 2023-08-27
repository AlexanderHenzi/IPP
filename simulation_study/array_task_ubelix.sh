#!/bin/bash
#SBATCH --mail-type=end,fail
#SBATCH --job-name="dr2"
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=1600M
#SBATCH --array=1-7000
#SBATCH --partition="epyc2"

#### Your shell commands below this line ####

module load R
R CMD BATCH --no-save --no-restore simulation_study.R

