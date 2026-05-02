#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 48:59:00
#SBATCH --partition=h100 --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#source /home/fako/venvs/mace_env/bin/activate
source /home/fako/venvs/cft_env/bin/activate

python run.py >> log.txt
