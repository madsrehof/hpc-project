#!/bin/bash
#BSUB -J simulate_parallel
#BSUB -q hpc
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
##BSUB -u mekre@dtu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
#BSUB -W 6:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonE5_2650v4]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# For profiling 
# python3 -m cProfile simulate.py 10

# For line profiling 
# kernprof -l simulate.py 10
# python3 -m line_profiler simulate.py.lprof

# Just for running
python3 simulate_parallel_static.py 64


