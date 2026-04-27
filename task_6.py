from os.path import join
import sys
import os

import numpy as np
from multiprocessing.pool import Pool
import time

import matplotlib.pyplot as plt

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE +2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    print("Multiprocessing using Dynamic scheduling:")
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    cwd = os.getcwd()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    if len(sys.argv) < 3:
        M = 1
    else:
        M = int(sys.argv[2])

    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    RUNS = 5
    NUM_PROCS = [i+1 for i in range(M)]

    def jacobi_partial(arg):
        u, mask = arg
        return jacobi(u, mask, MAX_ITER, ABS_TOL)
    
    arr = list(zip(all_u0, all_interior_mask))

    average_run_times = []
    for run in range(RUNS):
        run_times = []
        for num_procs in NUM_PROCS:

            start = time.time()
            with Pool(num_procs) as pool:
                results = list(pool.imap_unordered(jacobi_partial, arr, chunksize=1))
            run_times.append(time.time()-start)

        # Calculate average run times across runs using a running average formula
        average_run_times = [run * a/(run+1) + t/(run+1) for a, t in zip(average_run_times or [0] * len(run_times), run_times)]

    # Calculate speedups based on average run times
    speedups = [average_run_times[0]/average_run_times[i] for i in range(len(NUM_PROCS))]

    print("Average run times across runs:", average_run_times)
    print("Speedups:", speedups)

    fig, ax = plt.subplots()
    ax.set_title(f"Multicore with dynamic scheduling\n Runtimes on {N} floorplans", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("p: number of cores", fontsize=11)
    ax.set_ylabel("Wall clock time in seconds", fontsize=11)
    ax.plot(NUM_PROCS, average_run_times)
    fig.tight_layout()
    fig.savefig(cwd+"/plots/runtimes_dynamic")

    fig, ax = plt.subplots()
    ax.set_title(f"Multicore with dynamic scheduling\n Speedups on {N} floorplans", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("p: number of cores", fontsize=11)
    ax.set_ylabel("Speedups: S(p) = T(1)/T(p)", fontsize=11)
    ax.plot(NUM_PROCS, speedups)
    fig.tight_layout()
    fig.savefig(cwd+"/plots/speedups_dynamic")

