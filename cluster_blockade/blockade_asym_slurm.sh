#!/bin/bash
# SLURM alternative: one array task per n. Put this file, run_blockade.sh and
# blockade_asymmetric.py in ONE directory and from there:
#   mkdir -p logs && sbatch blockade_asym_slurm.sh
# then: python blockade_asymmetric.py --merge blockade_asym_out --output blockades_asymmetric_p.pkl
# Needs a Python env with numpy/scipy (activate it here); ARC is pip-installed if missing.
#SBATCH --job-name=blockade_asym
#SBATCH --array=40-99
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=logs/blockade_%a.out

# module load python/3.11   # or: source activate <env>
bash run_blockade.sh "$SLURM_ARRAY_TASK_ID" \
    --atom Cs --m1 0.5 --m2 1.5 --conv --outdir blockade_asym_out
