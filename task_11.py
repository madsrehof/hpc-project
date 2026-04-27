from os.path import join
import sys
import os

import numpy as np
from multiprocessing.pool import Pool
from time import time

from numba import jit, cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE +2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    u_new = np.empty_like(u)
    rows, cols = u.shape
    for i in range(max_iter):
        delta = 0.0
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if interior_mask[row - 1, col - 1]: # interior_mask is
                    u_new[row, col] = 0.25 * (u[row - 1, col] + u[row + 1, col] + u[row, col - 1] + u[row, col + 1])
                    d = abs(u[row, col] - u_new[row, col])
                    if d > delta:
                        delta = d
                else:
                    u_new[row, col] = u[row, col] # boundary condition
        u, u_new = u_new, u
        if delta < atol:
            break
    return u


@cuda.jit
def _jacobi_step_batched(u, u_new, interior_mask):
    j, i, h = cuda.grid(3)  # k indexes the floorplan
    rows, cols = u.shape[1], u.shape[2]
    if h < u.shape[0] and 1 <= i < rows - 1 and 1 <= j < cols - 1:
        if interior_mask[h, i - 1, j - 1]:
            u_new[h, i, j] = 0.25 * (u[h, i, j-1] + u[h, i, j+1] +
                                      u[h, i-1, j] + u[h, i+1, j])
        else:
            u_new[h, i, j] = u[h, i, j]


# This function runs the Jacobi iterations on the GPU. It initializes device arrays, launches the kernel in a loop, and then copies the final result back to the host.
def jacobi_cuda_batched(all_u, all_interior_mask, max_iter):
    u_d     = cuda.to_device(np.copy(all_u))
    u_new_d = cuda.to_device(np.copy(all_u))
    mask_d  = cuda.to_device(all_interior_mask)

    tpb = (16, 16, 4)  # j (cols), i (rows), k (floorplans) — 1024 threads per block
    bpg = (
        (all_u.shape[2] + tpb[0] - 1) // tpb[0],  # cols
        (all_u.shape[1] + tpb[1] - 1) // tpb[1],  # rows
        (all_u.shape[0] + tpb[2] - 1) // tpb[2],  # floorplans
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
        num_procs = 1
    else:
        num_procs = int(sys.argv[2])

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

    # Parallel JIT Implementation:

    # warmup jit
    # jacobi_jit(all_u0[0], all_interior_mask[0], 1, ABS_TOL)

    # def jacobi_partial(arg):
    #     u, mask = arg
    #     return jacobi_jit(u, mask, MAX_ITER, ABS_TOL)
    
    # arr = list(zip(all_u0, all_interior_mask))

    # start = time.time()
    # with Pool(num_procs) as pool:
    #     results = list(pool.imap_unordered(jacobi_partial, arr, chunksize=1))
    # elapsed = time.time() - start

    # with Pool(num_procs) as pool:
    #     # warm up all workers
    #     pool.map(jacobi_partial, [(all_u0[i], all_interior_mask[i]) for i in range(num_procs)])
    #     # now time the actual run
    #     start = time.time()
    #     results = list(pool.imap_unordered(jacobi_partial, arr, chunksize=1))
    #     elapsed = time.time() - start

    start = time()
    all_u = jacobi_cuda_batched(all_u0, all_interior_mask, MAX_ITER)
    elapsed = time() - start

    print("Task 11: Batched CUDA kernel implementation")
    print(f"Finished processing {N} floorplans using batched CUDA implementation. Time taken = {elapsed :.2f} seconds.")

    # print(f"Finished processing {N} floorplans using {num_procs} processors and JIT compilation. Time taken = {elapsed :.2f} seconds.")



