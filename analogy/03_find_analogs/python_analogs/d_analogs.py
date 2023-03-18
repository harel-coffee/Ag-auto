import numpy as np
import pandas as pd

main_in = "/data/hydro/users/Hossein/analog/"
main_out_dir = "/data/hydro/users/Hossein/analog/z_python_results/ensembles/"

us_features_dir = main_in + "usa/ready_features/"
all_data_usa = pd.read_csv(us_features_dir + "all_data_usa.csv") 

local_feat_main = main_in + "local/ready_features/"
avg_dir = local_feat_main + "averaged_data/"


"""
Append

- First append the needed treated unit to all historical/control units so we can find nearest locations.
- Then take numeric part of the data to be able to operate on them.
    - normalize
    - find distances
- add the distances back to the data frame.
- sort the data frame according to distances.
"""
rcp_types = ["rcp85"] # , "rcp45"

for rcp in rcp_types:
    avg = pd.read_csv(avg_dir + "averaged_data_" + rcp + ".csv") 
    all_data_usa.head(2)
    avg.head(2)
    out_dir = main_out_dir + rcp + "/"
    for ii in range(1, avg.shape[0]):

        one_site = avg.iloc[ii, :].copy()
        dt = all_data_usa.append(one_site)

        dt['year'] = dt['year'].apply(str);
        dt['treatment'] = dt['treatment'].apply(str);
        dt.head(2)
        dt_orig = dt.copy()

        # Normalize data
        mins = dt.min(axis=0, numeric_only=True)
        maxs = dt.max(axis=0, numeric_only=True)
        dt.iloc[:, 2:10]  = (dt.iloc[:, 2:10] - mins) / (maxs - mins)
        dt.head(2)
        dt.tail(2)

        """
        compute Euclidean distances
        Subtract the last row from all other rows to compute distances.
        """
        dt.iloc[:, 2:10] = dt.iloc[:, 2:10] - dt.iloc[-1, 2:10]
        print (dt.head(2))
        distances = (np.sqrt((dt.iloc[:, 2:10]**2).sum(axis=1))).values.reshape((-1,1))


        dt_orig['distances'] = distances
        dt_orig = dt_orig.sort_values(by=['distances'])
        dt_orig.head(3)

        out_name = one_site['location'] + '_' + str(one_site['year']) + ".csv"
        print (out_name)

        dt_orig.to_csv(out_dir + out_name, sep='\t')

