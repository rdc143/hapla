"""
Generate phenotypes from genotype data in binary PLINK format.

Usage:
python3 generatePheno.py --bfile example --causal 1000 --phenos 10 --out output example
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bfile",
	help="Genotype file in binary PLINK format")
parser.add_argument("-e", "--h2", type=float, default=0.5,
	help="Heritability of simulated traits (0.5)")
parser.add_argument("-c", "--causal", type=int, default=1000,
	help="Number of causal SNPs (1000)")
parser.add_argument("-p", "--phenos", type=int, default=1,
	help="Number of phenotypes to simulate (1)")
parser.add_argument("--binary", action="store_true",
	help="Binary phenotypes from liability threshold model")
parser.add_argument("--prevalence", type=float, default=0.1,
	help="Prevalence of trait (0.01)")
parser.add_argument("--alpha", type=float,
	help="Weighted frequency-based variance")
parser.add_argument("--multi-pops", type=int, nargs="+",
	help="Simulate population specific effects")
parser.add_argument("-o", "--out", default="pheno.generate",
	help="Prefix for output files")
args = parser.parse_args()

# Import numerical libraries
import numpy as np
from cyvcf2 import VCF
from math import ceil, sqrt
from scipy.stats import norm
from hapla import functions
from hapla import reader_cy

### Load data
print("\rLoading PLINK file...", end="")
	
# Finding length of .fam and .bim file
assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
n = functions.extract_length(f"{args.bfile}.fam")
m = functions.extract_length(f"{args.bfile}.bim")
fam = np.loadtxt(f"{args.bfile}.fam", usecols=[0,1], dtype=np.str_)

# Read .bed file
with open(f"{args.bfile}.bed", "rb") as bed:
	G_mat = np.fromfile(bed, dtype=np.uint8, offset=3)
B = ceil(n/4)
G_mat.shape = (m, B)
print(f"\rLoaded genotype data: {n} samples and {m} SNPs.")

### Simulate phenotypes
Y = np.zeros((n, args.phenos), dtype=float) # Phenotype matrix
Z = np.zeros((n, args.phenos), dtype=float) # Breeding values matrix
if args.multi_pops is not None:
	X = np.zeros(n)
	b_list = [0] + args.multi_pops + [n]

# Extract causals
G = np.zeros((args.causal, n), dtype=float) # Genotypes or haplotype clusters

# Sample causal SNPs
for p in range(args.phenos):
	# Environmental component
	E = np.random.normal(loc=0.0, scale=np.sqrt(1 - args.h2), size=n)
	E_var = np.var(E, ddof=0)

	# Sample causal loci
	c = np.sort(np.random.permutation(m)[:G.shape[0]]).astype(int)
	reader_cy.phenoPlink(G_mat, G, c)

	# Sample causal effects and estimate true PGS
	if args.multi_pops is None:
		if args.alpha is None:
			b = np.random.normal(loc=0.0, scale=sqrt(args.h2/float(G.shape[0])), \
				size=G.shape[0])
		else: # Frequency-based variance
			f = np.mean(G, axis=1)/2.0
			b = np.random.normal(loc=0.0, scale=np.power(f*(1-f), args.alpha*0.5), \
				size=G.shape[0])

		# Genetic contribution
		X = np.dot(G.T, b)
		X_scale = np.sqrt(args.h2)/np.std(X, ddof=0)
		X *= X_scale
		X -= np.mean(X)

		# Environmental contribution
		E_cov = np.cov(X, E, ddof=0)[0,1]
		E_scale = (np.sqrt(E_cov**2 + (1 - args.h2)*E_var) - E_cov)/E_var
		E *= E_scale
		E -= np.mean(E)
	else: # Simulate population specific causal effects
		for pop in range(len(b_list)-1):
			if args.alpha is None:
				b = np.random.normal(loc=0.0, scale=sqrt(args.h2/float(G.shape[0])), \
					size=G.shape[0])
			else: # Frequency-based variance
				b = np.random.normal(loc=0.0, scale=np.power(f*(1-f), args.alpha*0.5), \
					size=G.shape[0])
			
			# Genetic contribution
			X[b_list[pop]:b_list[pop+1]] = np.dot(G[:,b_list[pop]:b_list[pop+1]].T, b)
			X_scale = np.sqrt(args.h2)/np.std(X[b_list[pop]:b_list[pop+1]], ddof=0)
			X[b_list[pop]:b_list[pop+1]] *= X_scale
			X[b_list[pop]:b_list[pop+1]] -= np.mean(X[b_list[pop]:b_list[pop+1]])

			# Environmental contribution
			E_cov = np.cov(X[b_list[pop]:b_list[pop+1]], E[b_list[pop]:b_list[pop+1]], ddof=0)[0,1]
			E_scale = (np.sqrt(E_cov**2 + (1 - args.h2)*E_var) - E_cov)/E_var
			E[b_list[pop]:b_list[pop+1]] *= E_scale
			E[b_list[pop]:b_list[pop+1]] -= np.mean(E[b_list[pop]:b_list[pop+1]])

	# Generate phenotype
	Y[:,p] = X + E
	Z[:,p] = X

	# Use liability threshold model
	if args.binary:
		Y[:,p] = Y[:,p] > norm.ppf(1 - args.prevalence)

# Save phenotypes and breeding values
Y = np.hstack((fam, np.round(Y, 7)))
Z = np.hstack((fam, np.round(Z, 7)))
np.savetxt(f"{args.out}.pheno", Y, fmt="%s", delimiter=" ")
np.savetxt(f"{args.out}.breed", Z, fmt="%s", delimiter=" ")
print(f"Saved simulated phenotypes as {args.out}.pheno")
print(f"Saved simulated breeding values as {args.out}.breed")
