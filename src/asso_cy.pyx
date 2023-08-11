# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dgemv, ddot
from libc.math cimport sqrt

##### hapla - association testing #####
### hapla regress
# Setup standardized haplotype clusters for a block
cpdef void haplotypeStandard(unsigned char[:,::1] Z_mat, double[:,::1] Z, \
		unsigned char[::1] K_chr, int c):
	cdef:
		int n = Z.shape[1]
		int W = K_chr.shape[0]
		int b = 0
		int i, k, w
		double pi, sd
	for w in range(W):
		for k in range(K_chr[b]):
			pi = 0.0
			sd = 0.0
			for i in range(2*n):
				if Z_mat[c+w,i] == k:
					Z[b,i//2] += 1.0
					pi += 1.0
			pi /= <double>n
			for i in range(n):
				sd += (Z[b,i]-pi)*(Z[b,i]-pi)
			sd = sqrt(sd/(<double>(n-1)))
			for i in range(n):
				Z[b,i] = (Z[b,i] - pi)/sd
			b += 1

# Fast LOOCV using SciPy BLAS routines
cpdef void loocv(double[:,::1] L, double[:,::1] y_prs, double[::1] y_mse, \
		double[:,::1] H, double[::1] p, double[::1] y, double[::1] x, int r) nogil:
	cdef:
		char *trans = "T"
		int n = L.shape[0]
		int b = L.shape[1]
		int i1 = 1
		int i2 = 1
		int i
		double alpha = 1.0
		double beta = 0.0
		double *H0 = &H[0,0]
		double *x0 = &x[0]
		double *L0
		double h
	for i in range(n):
		L0 = &L[i,0]
		dgemv(trans, &b, &b, &alpha, H0, &b, L0, &i1, &beta, x0, &i2)
		h = ddot(&b, L0, &i1, x0, &i2)
		y_prs[r,i] = p[i] - h*(y[i] - p[i])/(1.0 - h)
		y_mse[r] += (y[i] - y_prs[r,i])*(y[i] - y_prs[r,i])

# LOCO prediction for LOOCV using SciPy BLAS routines
cpdef void loocvLOCO(double[:,::1] L, double[:,::1] y_chr, double[::1] y_hat, \
		double[:,::1] H, double[::1] p, double[::1] y, double[::1] a, double[::1] x) \
		nogil:
	cdef:
		char *trans = "T"
		int C = y_chr.shape[1]
		int n = L.shape[0]
		int b = L.shape[1]
		int i1 = 1
		int i2 = 1
		int c, i
		double alpha = 1.0
		double beta = 0.0
		double *H0 = &H[0,0]
		double *x0 = &x[0]
		double *L0
		double e, h
	for i in range(n):
		L0 = &L[i,0]
		dgemv(trans, &b, &b, &alpha, H0, &b, L0, &i1, &beta, x0, &i2)
		h = ddot(&b, L0, &i1, x0, &i2)
		e = y[i] - p[i]
		for c in range(C):
			y_chr[i,c] = y_hat[i] - L[i,c]*(a[c] - x[c]*e/(1.0 - h))

# LOCO prediction for K-fold CV
cpdef void haplotypeLOCO(double[:,::1] L, double[:,::1] E_hat, double[:,::1] y_chr, \
		double[::1] y_hat, unsigned char[::1] N_ind):
	cdef:
		int n = y_chr.shape[0]
		int C = y_chr.shape[1]
		int c, i
	for i in range(n):
		for c in range(C):
			y_chr[i,c] = y_hat[i] - L[i,c]*E_hat[N_ind[i],c]



### hapla asso
# Setup haplotype clusters for a block and estimate frequencies
cpdef void haplotypeAssoc(unsigned char[:,::1] Z_mat, double[:,::1] Z, \
		double[:,::1] P, int B, int w):
	cdef:
		int K = Z.shape[0]
		int n = Z.shape[1]
		int i, k
	for k in range(K):
		for i in range(2*n):
			if Z_mat[w,i] == k:
				Z[k,i//2] += 1.0
				P[B+k,3] += 1.0
		P[B+k,3] /= <double>n

# Convert 1-bit into genotype block
cpdef void genotypeAssoc(unsigned char[:,::1] G_mat, double[:,::1] G, \
		double[:,::1] P, int B_idx):
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int B = G_mat.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = G_mat[B_idx+j,b]
			for bit in range(0, 8, 2):
				G[j,i] = byte & mask
				byte = byte >> 1 # Right shift 1 bit
				G[j,i] += byte & mask
				byte = byte >> 1 # Right shift 1 bit
				P[B_idx+j,2] += G[j,i]
				i = i + 1
				if i == n:
					break
		P[B_idx+j,2] /= 2.0*(<double>n)

# Association testing of haplotype cluster alleles
cpdef void haplotypeTest(double[:,::1] Z, double[:,::1] P, double[::1] y_res, \
		double s_env, int B, int w):
	cdef:
		int K = Z.shape[0]
		int n = Z.shape[1]
		int i, k
		double gTg, gTy
	for k in range(K):
		gTg = 0.0
		gTy = 0.0
		for i in range(n):
			gTg += Z[k,i]*Z[k,i]
			gTy += Z[k,i]*y_res[i]
		P[B+k,1] = w+1 # Window
		P[B+k,2] = k+1 # Cluster
		P[B+k,4] = gTy/gTg # Beta
		P[B+k,6] = gTy/(s_env*sqrt(gTg)) # Wald's
		P[B+k,5] = P[B+k,4]/P[B+k,6] # SE(Beta)
		P[B+k,6] *= P[B+k,6]

# Association testing of SNPs
cpdef void genotypeTest(double[:,::1] G, double[:,::1] P, double[::1] y_res, \
		double s_env, int B_idx):
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int i, j
		double gTg, gTy
	for j in range(m):
		gTg = 0.0
		gTy = 0.0
		for i in range(n):
			gTg += G[j,i]*G[j,i]
			gTy += G[j,i]*y_res[i]
		P[B_idx+j,3] = gTy/gTg # Beta
		P[B_idx+j,5] = gTy/(s_env*sqrt(gTg)) # Wald's
		P[B_idx+j,4] = P[B_idx+j,3]/P[B_idx+j,5] # SE(Beta)
		P[B_idx+j,5] *= P[B_idx+j,5]
