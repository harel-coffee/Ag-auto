library(stats)
library(data.frame)
library(dplyr)
library(matrixcalc)
library(Matrix)



compute_eros_distance_MTS <- function(MTS_1, MTS_2, weights){
	dist <- sqrt( 2 - 2 * compute_eros_similarity(MTS_1, MTS_2, weights))
	return (dist)
}

compute_eros_similarity_MTS <- function(MTS_1, MTS_2, weights){
	# compute covariances of the two MTS's
	# we need their rank and their SVD to compute similarities
	covariances <- compute_cov_matrices(MTS_1, MTS_1)
	cov_MTS_1 <- covariances[[1]]
	cov_MTS_2 <- covariances[[2]]

	# Rank of the covariance matrices
	# is used as upper bound for the summation.
	rank_1 <- rankMatrix(cov_MTS_1)[1]
	rank_2 <- rankMatrix(cov_MTS_2)[1]
	min_rank <- min(rank_1, rank_2)
	
	SVD_MTS_1 <- svd(cov_MTS_1, nu=0)
	SVD_MTS_2 <- svd(cov_MTS_2, nu=0)

	right_eigvec_1 <- SVD_MTS_1[[2]]
	singulars_1 <- SVD_MTS_1[[1]]

	right_eigvec_2 <- SVD_MTS_2[[2]]
	singulars_2 <- SVD_MTS_2[[1]]

    right_eigvec_1 <- right_eigvec_1[, 1:min_rank]
    right_eigvec_2 <- right_eigvec_2[, 1:min_rank]
    
    inner_eig <- abs(colSums(right_eigvec_1 * right_eigvec_2))
    ##
    ## similarity_measure \in [0, 1]
    ## 1 is the most similar
    ##
    similarity_measure <- sum(weights * inner_eig)
    return (similarity_measure)
}

compute_normalized_weights_MTS <- function(M, method="mean"){
	# input : M is a matrix of size (N, n)
	#           N: no. of MTS sets in the data set
	#           n: no. of variables
	# **** Please note that I use a matrix that is
	#      transpose of the one used in the original PCA paper!!!    
	# output: weights computed by normalized eigenvectors.
	M <- M / rowSums(M)
	weights <- compute_raw_weights(M, method="mean")
	return (weights)
}

compute_raw_weights_MTS <- function(M, method="mean"){
	# input : M is a matrix of size (N, n)
	#           N: no. of MTS sets in the data set
	#           n: no. of variables
	# **** Please note that I use a matrix that is
	#      transpose of the one used in the original PCA paper!!!    
	# output: weights computed by raw eigenvectors.
    
	M = data.table(M)
	if (method=="mean"){
		weights <- M[, lapply(.SD, mean)]
		weights <- weights/sum(weights)
		} elseif(method=="min"){
			weights <- M[, lapply(.SD, min)]
			weights <- weights/sum(weights)
		}elseif(method=="max"){
			weights <- M[, lapply(.SD, max)]
			weights <- weights/sum(weights)
		}
	return(weights)
}

compute_cov_SVDs_MTS <- function(M_1, M_2){
	# input:  two multivariate time series data tables
	# output: SVD decomposition of covariance matrix
	#         of the inputs (only singular values and right eigenvectors)
	#         Recall: singular values of this decompisition 
	#                 are eigenvalues of the MTS's squared.
	# 
	covariances <- compute_cov_matrices(M_1, M_2)
	M_1_cov <- covariances[[1]]
	M_2_cov <- covariances[[2]]
	M_1_SVD <- svd(M_1_cov, nu=0)
	M_2_SVD <- svd(M_2_cov, nu=0)
    return(M_1_SVD, M_2_SVD)
}

compute_cov_matrices_MTS <- function(MTS_1, MTS_2){
	# input : MTS_1, MTS_2, two multivariate time series (MTS) matrices.
	#         Recall: cov automatically shifts the columns by their means! (we good)
	# output: the covariance of the two MTS.
	MTS_1_cov = cov(MTS_1)
	MTS_2_cov = cov(MTS_2)
	return (list(MTS_1_cov, MTS_2_cov))
    # The way to use the output of this function
	# RelData = data.table(output[[1]])
}

