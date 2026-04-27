from os.path import join
import sys
import os
from time import time

import numpy as np

from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE +2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# Reference Jacobi function
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


# This is the kernel that performs a single Jacobi iteration step. Each thread computes one grid point in the output array `u_new`.
@cuda.jit
def _jacobi_step(u, u_new, interior_mask):
    j, i = cuda.grid(2)
    rows, cols = u.shape
    if 1 <= i < rows - 1 and 1 <= j < cols - 1:
        if interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (u[i, j-1] + u[i, j+1] +
                                   u[i-1, j] + u[i+1, j])
        else:
            u_new[i, j] = u[i, j]


# This function runs the Jacobi iterations on the GPU. It initializes device arrays, launches the kernel in a loop, and then copies the final result back to the host.
def jacobi_cuda(u, interior_mask, max_iter):
    # move everything to the GPU
    u_d     = cuda.to_device(np.copy(u))
    u_new_d = cuda.to_device(np.copy(u))
    mask_d  = cuda.to_device(interior_mask)

    tpb = (32, 32)
    bpg = (
        (u.shape[0] + tpb[0] - 1) // tpb[0],
        (u.shape[1] + tpb[1] - 1) // tpb[1],
    )

    for i in range(max_iter):
        _jacobi_step[bpg, tpb](u_d, u_new_d, mask_d)
        u_d, u_new_d = u_new_d, u_d          # ping-pong buffers to avoid unnecessary copying

    return u_d.copy_to_host() # copy result back to host and return it


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
    print("Running Jacobi iterations on GPU...")
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

    all_u = np.empty_like(all_u0)

    start = time()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u
    elapsed = time() - start
    print(f"Total elapsed time: {elapsed:.2f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys)) # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))