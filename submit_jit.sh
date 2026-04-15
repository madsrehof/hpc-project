#!/bin/bash
#BSUB -J simulate_jit
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -B
#BSUB -N
##BSUB -u mekre@dtu.dk
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Output_%J.err
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonE5_2650v4]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python3 simulate_jit.py 100 



