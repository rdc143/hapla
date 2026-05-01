"""
hapla.
Evaluate fit of admix model by computing correlations of residuals.
"""

__author__ = "Thomas Bøggild"

# Libraries
import os
from datetime import datetime
from time import time
from hapla import __version__


##### hapla eval #####
def main(args, deaf):
    print("-----------------------------------")
    print(f"hapla by Jonas Meisner (v{__version__})")
    print(f"hapla eval using {args.threads} thread(s)")
    print("-----------------------------------\n")

    # Check input
    assert (args.filelist is not None) or (args.clusters is not None), (
        "No input data (--filelist or --clusters)!"
    )
    assert args.threads > 0, "Please select a valid number of threads!"
    assert args.qfile is not None, "No Q file provided (--qfile)!"

    start = time()

    # Control threads of external numerical libraries
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_MAX_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_MAX_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

    # Create log-file of used arguments
    full = vars(args)
    with open(f"{args.out}.log", "w") as log:
        log.write(f"hapla v{__version__}\n")
        log.write("hapla eval\n")
        log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        log.write(f"Directory: {os.getcwd()}\n")
        log.write("Options:\n")
        for key in full:
            if full[key] != deaf[key]:
                log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(
                    f"\t--{key} {full[key]}\n"
                )
    del full, deaf

    # Import numerical libraries and cython functions
    import numpy as np
    from hapla import eval_cy

    # Prepare list of data files
    if args.filelist is not None:
        Z_list = []  # List of filenames
        with open(args.filelist) as f:
            for z_file in f:
                # Check input across files and count windows
                z = z_file.strip("\n")
                Z_list.append(z)
                assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
                assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
                assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
                if len(Z_list) == 1:  # First file
                    z_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
                    k_vec = np.genfromtxt(
                        f"{z}.win", dtype=np.uint32, usecols=[5]
                    ).reshape(-1)
                    N = z_ids.shape[0]
                    w_list = [k_vec.shape[0]]
                else:  # Loop files
                    t_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
                    assert np.sum(z_ids != t_ids) == 0, (
                        "Samples do not match across files!"
                    )
                    k_tmp = np.genfromtxt(
                        f"{z}.win", dtype=np.uint32, usecols=[5]
                    ).reshape(-1)
                    k_vec = np.append(k_vec, k_tmp)
                    w_list.append(k_tmp.shape[0])
        F = len(Z_list)
        w_vec = np.array(w_list, dtype=np.uint32)
        del z_ids, w_list
    else:  # Single file (chromosome)
        F = 1
        Z_list = [args.clusters]
        assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
        k_vec = np.genfromtxt(
            f"{Z_list[0]}.win", dtype=np.uint32, usecols=[5]
        ).reshape(-1)
        w_vec = np.array([k_vec.shape[0]], dtype=np.uint32)
        N = np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_).shape[0]
    print(f"Parsing {F} file(s).")

    # Load Q matrix
    Q = np.ascontiguousarray(np.genfromtxt(args.qfile, dtype=float))
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    assert Q.shape[0] == N, "Number of samples doesn't match!"
    assert Q.shape[1] > 1, "Please provide at least two ancestral components!"
    A = np.ascontiguousarray(np.linalg.pinv(np.dot(Q.T, Q)))

    # Covariance containers
    C = np.zeros((N, N), dtype=np.float64)
    V = np.zeros(N, dtype=np.float64)

    # Loop over chromosomes
    print("Computing correlations of residuals.")
    w_cnt = 0
    for z in np.arange(F):  # Loop through files
        print(f"Processing file {z + 1}/{F}")
        t_chr = time()
        W_chr = w_vec[z]
        k_chr = k_vec[w_cnt : (w_cnt + W_chr)]
        w_cnt += W_chr

        # Load haplotype cluster assignment file
        with open(f"{Z_list[z]}.bca", "rb") as f:
            # Check magic numbers
            magic = np.fromfile(f, dtype=np.uint8, count=3)
            assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), (
                "Magic number doesn't match file format!"
            )

            # Add haplotype cluster assignments to container
            Z_chr = np.fromfile(f, dtype=np.uint8)
            Z_chr.shape = (W_chr, 2 * N)
        assert np.max(k_chr) == (np.max(Z_chr) + 1), "Number of clusters doesn't match!"

        # Project cluster counts onto the ADMIXTURE Q-space and accumulate residual covariances.
        eval_cy.covar(C, V, Q, A, Z_chr, k_chr)

        if F > 1:
            # Print elapsed time of chromosome
            t_tmp = time() - t_chr
            t_min = int(t_tmp // 60)
            t_sec = int(t_tmp - t_min * 60)
            print(f"Elapsed time: {t_min}m{t_sec}s\n")
    # Estimate model-expected covariance of residuals and convert into correlations
    S = np.dot(Q.T, V.reshape(-1, 1) * Q)
    B = np.ascontiguousarray(np.dot(A, np.dot(S, A)))
    QA = np.ascontiguousarray(np.dot(Q, A))
    QB = np.ascontiguousarray(np.dot(Q, B))
    C_exp = np.zeros_like(C)
    eval_cy.expected(C_exp, Q, QA, QB, V)
    b_hat = np.zeros_like(C)
    c_hat = np.zeros_like(C)
    eval_cy.corr(C, b_hat)
    eval_cy.corr(C_exp, c_hat)
    cor = b_hat - c_hat

    # Write correlations to files
    np.savetxt(f"{args.out}.bhat", b_hat, fmt="%.4f")
    np.savetxt(f"{args.out}.chat", c_hat, fmt="%.4f")
    np.savetxt(f"{args.out}.corres", cor, fmt="%.4f")

    # Clean
    del C, C_exp, b_hat, c_hat, cor

    # Print elapsed time for computation
    t_tot = time() - start
    t_min = int(t_tot // 60)
    t_sec = int(t_tot - t_min * 60)
    print(f"Total elapsed time: {t_min}m{t_sec}s")

    # Write to log-file
    with open(f"{args.out}.log", "a") as log:
        log.write(f"\nSaved empirical correlations of residuals as {args.out}.bhat\n")
        log.write(f"Saved model-expected correlations of residuals as {args.out}.chat\n")
        log.write(f"Saved corrected correlations of residuals as {args.out}.corres\n")
        log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")


##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla eval' command!"
