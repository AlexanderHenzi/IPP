#!/bin/bash
#SBATCH --mail-type=end,fail
#SBATCH --job-name="independence"
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=epyc2
#SBATCH --array=1-1000

#### Your shell commands below this line ####

module load r/4.1.3
# module load python/?
R CMD BATCH --no-save --no-restore simulation_study_2_computations.R