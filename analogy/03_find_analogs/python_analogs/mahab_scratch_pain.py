def one_location_dist(curr_location_df, complete_hist_df, numeric_feat, NN_count):
    """
    input: curr_location_df: data frame of current location, including one location, all future years
           complete_hist_df: historical data frame (analog pool)
           numeric_feat: list of column names that are numeric
           NN_count: number of NNs we want
    
    output: two data frames:
                includes list of years and locations of nearest neighbors
                includes list of distaces of NNs to the queries.
    """

    # initiatae data frames to attach the locations that are NNs
    NNs_df   = curr_location_df[['year', 'location']].copy() # data frame containing (year, location)
    dists_df = curr_location_df[['year', 'location']].copy() # data frame containing distances
    """
    ## Make the ICV
    Copy all the historical data into `ICV` so they are separate
    and we are clear on what is going on

    `ICV` is used to remove inter anual variability! 
    This is the one we have to get covariance matrix from.
    """
    
    # form the ICV to compute its covariance to remove inter-annual variability
    ICV = complete_hist_df.copy()
    ICV = ICV.loc[ICV['location'] == curr_location_df.location.unique()[0]] # filter corresponding location
    #############################################################################
    #
    #          Normalize before doing anything
    #
    #############################################################################
    ICV_means = curr_location_df.loc[:, numeric_feat].mean()
    ICV_stds = curr_location_df.loc[:, numeric_feat].std()
    
    ICV = (ICV.loc[:, numeric_feat] - ICV_means) / ICV_stds
    curr_location_df = (curr_location_df.loc[:, numeric_feat] - ICV_means) / ICV_stds
    complete_hist_df = (complete_hist_df.loc[:, numeric_feat] - ICV_means) / ICV_stds
    
    #
    # pick numerical part of the data frame to do the operations:
    #
    complete_hist_df_numeric = complete_hist_df.loc[:, numeric_feat].copy()
    future_numeric = curr_location_df.loc[:, numeric_feat].copy()
    ICV = ICV.loc[:, numeric_feat]
    
    ### Apply PCA here and use those to find analogs
    pca = PCA(n_components = detect_effective_compon(ICV))
    pca.fit(ICV)
    #
    # transform data into PCA space to compute analogs
    ICV_pca = pca.transform(ICV)
    hist_pca = pca.transform(complete_hist_df_numeric)
    future_pca = pca.transform(future_numeric)
    
    # compute covariance of ICV_pca
    # the robust thing changes every time! is it based on a random start
    # of an interative method?
    """
    robust_cov = MinCovDet().fit(ICV_pca)
    robust_cov = robust_cov.covariance_
    robust_cov_inv = np.linalg.inv(robust_cov)
    # np.cov(ICV_pca);
    """
    # the following is the same as [(1/N) * np.matmul(M.transpose(), M)]. which is not even divided by N-1
    cov = sklearn.covariance.empirical_covariance(ICV_pca, assume_centered=False)
    cov_inv = np.linalg.inv(cov)
    ##
    ## Find nearest neighbors
    ##
    future_yr_count = len(curr_location_df.year.unique())
    
    for yr in np.arange(future_yr_count):
        # list of years and locations in historical data to use to attach the distances to
        hist_loc_year_frame = complete_hist_df[['year', 'location']].copy()
        
        curr_ft = future_pca[yr, ]
        curr_dists = one_sample_dist(curr_future=curr_ft, hist_dt=hist_pca, conar_inv=cov_inv)
       
        # add the distanced to the year_location data frame, 
        # so we know how far each point is from the query.
        hist_loc_year_frame['distance'] = curr_distance
        hist_loc_year_frame = hist_loc_year_frame.sort_values(by="distance") 
        hist_loc_year_frame = hist_loc_year_frame.iloc[0: NN_count,] # grab needed number of nearest neighbors
        
        
###################################################################################
#                                                                                 #
#        compute distance of one sample point to all points in analog pool        #
#                                                                                 #
###################################################################################
def one_sample_dist(curr_future, hist_dt, conar_inv):
    """
    inputs here are of np.ndarray type that are projections into PCA space
    
    inputs: curr_future: future data for one location, one year (a vector)
            hist_dt: historical data to find analogs in
            conar_inv: inverse of covariance matrix for M. distance

    output: list of distances of the given sample, (one_loc, one_year),
            from all historical samples. 
            (1293 locations * 37 years = 47841 distances)
    """
    diff_matrix = curr_future - hist_dt
    square_dists_matrix = np.matmul(diff_matrix, np.matmul(conar_inv, diff_matrix.transpose()))

    # take diagonal entries which are distances^2, and then take the sqrt.
    distances = np.sqrt(np.diagonal(square_dists_matrix))
    return (distances)