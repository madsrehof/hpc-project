#!/bin/bash
#BSUB -J Task_11
#BSUB -q c02613
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
##BSUB -u mekre@dtu.dk
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Output_%J.err
#BSUB -W 15
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6326]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python3 task_11.py 100


