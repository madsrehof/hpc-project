#!/bin/bash
#BSUB -J task_12
#BSUB -q gpua100
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
##BSUB -u mekre@dtu.dk
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Output_%J.err
#BSUB -W 30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python3 task_12.py 4571
