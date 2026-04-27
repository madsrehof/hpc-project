#!/bin/bash
#BSUB -J Task_5
#BSUB -q hpc
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
##BSUB -u mekre@dtu.dk
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Output_%J.err
#BSUB -W 6:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonE5_2650v4]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python3 simulate_parallel_static.py 100 16


