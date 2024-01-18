"""
Main caller of hapla.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import sys

# Main function
def main():
	# Argparser
	parser = argparse.ArgumentParser(prog="hapla")
	subparsers = parser.add_subparsers(title="hapla commands")

	### Commands
	# hapla cluster
	parser_cluster = subparsers.add_parser("cluster")
	parser_cluster.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_cluster.add_argument("-w", "--win", type=int, default=8,
		metavar="INT", help="Window size (8)")
	parser_cluster.add_argument("-l", "--lmbda", type=float, default=0.1,
		metavar="FLOAT", help="Set lambda hyperparameter (0.1)")
	parser_cluster.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_cluster.add_argument("-o", "--out", default="hapla.cluster",
		metavar="OUTPUT", help="Output prefix")
	parser_cluster.add_argument("--min-mac", type=int, default=10,
		metavar="INT", help="Minimum allele count for haplotype cluster (10)")
	parser_cluster.add_argument("--max-clusters", type=int, default=128,
		metavar="INT", help="Maximum number of haplotype clusters per window (128)")
	parser_cluster.add_argument("--max-iterations", type=int, default=500,
		metavar="INT", help="Maximum number of iterations (500)")
	parser_cluster.add_argument("--medians", action="store_true",
		help="Save haplotype cluster medians")
	parser_cluster.add_argument("--loglike", action="store_true",
		help="Compute log-likelihoods for ancestry estimation")
	parser_cluster.add_argument("--plink", action="store_true",
		help="Generate binary PLINK output")
	parser_cluster.add_argument("--duplicate-fid", action="store_true",
		help="Use sample list as family ID (PLINK 1.9 compatibility)")
	parser_cluster.add_argument("--overlap", action="store_true",
		help="Haplotype clustering in overlapping windows")
	parser_cluster.add_argument("--num-overlap", type=int,
		metavar="INT", help="Number of overlapping windows")
	parser_cluster.add_argument("--verbose", action="store_true",
		help="Verbose output from each iteration")

	# hapla struct
	parser_struct = subparsers.add_parser("struct")
	parser_struct.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster alleles files")
	parser_struct.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster alleles file")
	parser_struct.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_struct.add_argument("-o", "--out", default="hapla.pca",
		metavar="OUTPUT", help="Output prefix")
	parser_struct.add_argument("--min-freq", type=float,
		metavar="FLOAT", help="Minimum frequency for haplotype cluster")
	parser_struct.add_argument("--grm", action="store_true",
		help="Estimate genome-wide relationship matrix (GRM)")
	parser_struct.add_argument("--alpha", type=float, default=0.0,
		metavar="FLOAT", help="Alpha scaling parameter in GRM (0.0)")
	parser_struct.add_argument("--no-centering", action="store_true",
		help="Do not perform Gower and data centering on GRM")
	parser_struct.add_argument("--iid", metavar="FILE",
		help="Sample ID list for GCTA format")
	parser_struct.add_argument("--fid", metavar="FILE",
		help="Family ID list for GCTA format")
	parser_struct.add_argument("--pca", type=int,
		metavar="INT", help="Perform PCA and extract eigenvectors")
	parser_struct.add_argument("--loadings", action="store_true",
		help="Save loadings of SVD")
	parser_struct.add_argument("--randomized", action="store_true",
		help="Use randomized SVD (for very large sample sizes)")
	parser_struct.add_argument("--batch", type=int, default=4096,
		metavar="INT", help="Number of clusters in batched SVD")

	# hapla predict
	parser_predict = subparsers.add_parser("predict")
	parser_predict.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_predict.add_argument("-m", "--medians", metavar="FILE",
		help="Input haplotype cluster medians as binary NumPy array")
	parser_predict.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_predict.add_argument("-o", "--out", default="hapla.predict",
		metavar="OUTPUT", help="Output prefix")
	parser_predict.add_argument("--plink", action="store_true",
		help="Generate binary PLINK output")
	parser_predict.add_argument("--duplicate-fid", action="store_true",
		help="Use sample list as family ID (PLINK 1.9 compatibility)")

	# Parse arguments
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()

	### Run specified command
	# hapla cluster
	if sys.argv[1] == "cluster":
		if len(sys.argv) < 3:
			parser_cluster.print_help()
			sys.exit()
		else:
			from hapla import cluster
			cluster.main(args)
	
	# hapla struct
	if sys.argv[1] == "struct":
		if len(sys.argv) < 3:
			parser_struct.print_help()
			sys.exit()
		else:
			from hapla import struct
			struct.main(args)

	# hapla predict
	if sys.argv[1] == "predict":
		if len(sys.argv) < 3:
			parser_predict.print_help()
			sys.exit()
		else:
			from hapla import predict
			predict.main(args)



##### Define main #####
if __name__ == "__main__":
	main()
