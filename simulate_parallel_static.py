from os.path import join
import sys
import os

import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool, Pool
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
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
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
    NUM_PROCS = [1, 2, 4, 8, 16, 32]

    # all_u = np.empty_like(all_u0)
    # for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
    #     u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    #     all_u[i] = u

    def jacobi_partial(arg):
        u, mask = arg
        return jacobi(u, mask, MAX_ITER, ABS_TOL)
    
    arr = list(zip(all_u0, all_interior_mask))

    times = []
    for num_procs in NUM_PROCS:
        chunksize = N/num_procs
        start = time.time()
        with Pool(num_procs) as pool:
            all_u = pool.map(jacobi_partial, arr)
        end = time.time()
        times.append(end-start)
        print(f"Finished processing {N} floorplans using {num_procs} processor(s). Time taken = {end-start}")

    fig, ax = plt.subplots()
    ax.set_title(f"Multicore with static scheduling runtimes on {N} floorplans", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("p: number of cores", fontsize=11)
    ax.set_ylabel("Wall clock time in seconds", fontsize=11)
    ax.plot([1, 2, 4, 8, 16, 32], times)
    cwd = os.getcwd()
    fig.savefig(cwd+"/plots/runtimes_static")

    speedups = [times[0]/times[i] for i in range(len(NUM_PROCS))]

    fig, ax = plt.subplots()
    ax.set_title(f"Multicore with static scheduling speedups on {N} floorplans", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("p: number of cores", fontsize=11)
    ax.set_ylabel("Speedups: S(p) = T(1)/T(p)", fontsize=11)
    ax.plot([1, 2, 4, 8, 16, 32], speedups)
    cwd = os.getcwd()
    fig.savefig(cwd+"/plots/speedups_static")


    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys)) # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))