from os.path import join
import sys
import os

import numpy as np
from time import time

from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def _jacobi_step_batched(u, u_new, interior_mask):
    j, i, h = cuda.grid(3)
    rows, cols = u.shape[1], u.shape[2]
    if h < u.shape[0] and 1 <= i < rows - 1 and 1 <= j < cols - 1:
        if interior_mask[h, i - 1, j - 1]:
            u_new[h, i, j] = 0.25 * (u[h, i, j-1] + u[h, i, j+1] +
                                      u[h, i-1, j] + u[h, i+1, j])
        else:
            u_new[h, i, j] = u[h, i, j]


def jacobi_cuda_batched(all_u, all_interior_mask, max_iter):
    u_d     = cuda.to_device(np.copy(all_u))
    u_new_d = cuda.to_device(np.copy(all_u))
    mask_d  = cuda.to_device(all_interior_mask)

    tpb = (16, 16, 4)
    bpg = (
        (all_u.shape[2] + tpb[0] - 1) // tpb[0],
        (all_u.shape[1] + tpb[1] - 1) // tpb[1],
        (all_u.shape[0] + tpb[2] - 1) // tpb[2],
    )

    for _ in range(max_iter):
        _jacobi_step_batched[bpg, tpb](u_d, u_new_d, mask_d)
        u_d, u_new_d = u_new_d, u_d

    return u_d.copy_to_host()


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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = len(all_building_ids)
    else:
        N = int(sys.argv[1])

    # Job array index (1-based), determines which half to process
    if len(sys.argv) < 3:
        job_index = 0
    else:
        job_index = int(sys.argv[2]) - 1  # convert to 0-based

    all_building_ids = all_building_ids[:N]
    # half = N // 2
    # start_idx = job_index * half
    # end_idx = N if job_index == 1 else half
    # building_ids = all_building_ids[start_idx:end_idx]

    n = len(all_building_ids)

    # Load floor plans
    all_u0 = np.empty((n, 514, 514))
    all_interior_mask = np.empty((n, 512, 512), dtype='bool')
    for i, bid in enumerate(all_building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000

    start = time()
    all_u = jacobi_cuda_batched(all_u0, all_interior_mask, MAX_ITER)
    elapsed = time() - start

    print("Job started at time {:.2f} seconds.".format(start), flush=True)
    print("Job finished at time {:.2f} seconds.".format(start + elapsed), flush=True)

    print(f"Job {job_index + 1}: processed {n} floorplans in {elapsed:.2f} seconds.", flush=True)

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys), flush=True)
    for bid, u, interior_mask in zip(all_building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys), flush=True)
