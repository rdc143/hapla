"""
hapla.
Evaluate fit of admix model by computing correlations of residuals as in evalAdmix.
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
    assert (args.pfilelist is not None) or (args.pfile is not None), (
        "No P file(s) provided (--pfilelist or --pfile)!"
    )
    if args.filelist is not None:
        assert args.pfilelist is not None, "Input formats don't match!"
    if args.clusters is not None:
        assert args.pfile is not None, "Input formats don't match!"
    assert args.qfile is not None, "No Q file provided (--qfile)!"
    assert args.threads > 0, "Please select a valid number of threads!"

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
    from hapla import functions
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
                    w_tmp = np.genfromtxt(f"{z}.win", dtype=int, usecols=[1, 2, 5])
                    k_vec = w_tmp[:, 2].astype(np.uint32)
                    N = 2 * z_ids.shape[0]
                    w_list = [k_vec.shape[0]]
                else:  # Loop files
                    t_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
                    assert np.sum(z_ids != t_ids) == 0, (
                        "Samples do not match across files!"
                    )
                    w_tmp = np.genfromtxt(f"{z}.win", dtype=int, usecols=[1, 2, 5])
                    k_vec = np.append(k_vec, w_tmp[:, 2].astype(np.uint32))
                    w_list.append(w_tmp.shape[0])
        F = len(Z_list)
        w_vec = np.array(w_list, dtype=np.uint32)
        del z_ids, t_ids, w_tmp, w_list
    else:  # Single file (chromosome)
        F = 1
        Z_list = [args.clusters]
        assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
        w_tmp = np.genfromtxt(f"{Z_list[0]}.win", dtype=int, usecols=[1, 2, 5])
        k_vec = w_tmp[:, 2].astype(np.uint32)
        w_vec = np.array([k_vec.shape[0]], dtype=np.uint32)
        N = 2 * np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_).shape[0]
        del w_tmp
    print(f"Parsing {F} file(s).")

    # Load Q matrix
    Q = np.genfromtxt(args.qfile, dtype=float)
    Q = np.repeat(Q, 2, axis=0)
    assert Q.shape[0] == N, "Number of samples doesn't match!"
    K = Q.shape[1]

    # Prepare list of P files
    if F > 1:
        w_cnt = 0
        P_list = []
        with open(args.pfilelist) as f:
            for p, p_file in enumerate(f):
                assert os.path.isfile(p_file.strip("\n")), (
                    f"The {p + 1}/{F} matrix file doesn't exist!"
                )
                P_list.append(p_file.strip("\n"))
        assert len(P_list) == F, "Number of files doesn't match!"

    # Covariance container
    N_ind = N // 2
    C = np.zeros((N_ind, N_ind), dtype=np.float64)

    # Loop over chromosomes
    print(f"Computing correlations of residuals")
    for z in np.arange(F):  # Loop through files
        print(f"Processing file {z + 1}/{F}")
        t_chr = time()
        W_chr = w_vec[z]

        # Load P matrix file
        if F > 1:
            P_chr = np.genfromtxt(P_list[z], dtype=float).reshape(-1)
            k_chr = k_vec[w_cnt : (w_cnt + W_chr)]
            w_cnt += W_chr
        else:
            P_chr = np.genfromtxt(args.pfile, dtype=float).reshape(-1)
            k_chr = k_vec
        c_chr = np.insert(np.cumsum(k_chr * K, dtype=np.uint32), 0, 0)
        L_chr = np.sum(k_chr, dtype=int)
        assert P_chr.shape[0] == (L_chr * K), "Number of clusters doesn't match!"

        # Load haplotype cluster assignment file
        with open(f"{Z_list[z]}.bca", "rb") as f:
            # Check magic numbers
            magic = np.fromfile(f, dtype=np.uint8, count=3)
            assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), (
                "Magic number doesn't match file format!"
            )

            # Add haplotype cluster assignments to container
            Z_chr = np.fromfile(f, dtype=np.uint8)
            Z_chr.shape = (W_chr, N)
        assert np.max(k_chr) == (np.max(Z_chr) + 1), "Number of clusters doesn't match!"

        # Convert assignments to one-hot encoding one window at a time, compute expected assignments,
        # then residuals and lastly covariances
        eval_cy.covar(C, Q, P_chr, Z_chr, k_chr, c_chr, K)
        
        if F > 1:
            # Print elapsed time of chromosome
            t_tmp = time() - t_chr
            t_min = int(t_tmp // 60)
            t_sec = int(t_tmp - t_min * 60)
            print(f"Elapsed time: {t_min}m{t_sec}s\n")

    
    # Sum covariances over chromosomes and convert into correlations
    cor = np.zeros((N_ind, N_ind), dtype=np.float64)
    eval_cy.corr(C, cor)
    
    # Write correlations to file
    np.savetxt(f"{args.out}.corres", cor, fmt="%.4f")

    # Clean
    del C, cor

    # Print elapsed time for computation
    t_tot = time() - start
    t_min = int(t_tot // 60)
    t_sec = int(t_tot - t_min * 60)
    print(f"Total elapsed time: {t_min}m{t_sec}s")

    # Write to log-file
    with open(f"{args.out}.log", "a") as log:
        log.write(f"\nSaved correlations of residuals as {args.out}.corres\n")
        log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")


##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla eval' command!"
