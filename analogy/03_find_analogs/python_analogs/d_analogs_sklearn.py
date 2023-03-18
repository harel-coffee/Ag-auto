import numpy as np
import pandas as pd
from sklearn import preprocessing

main_in = "/data/hydro/users/Hossein/analog/"

us_features_dir = main_in + "usa/ready_features/"
all_data_usa = pd.read_csv(us_features_dir + "all_data_usa.csv") 

local_feat_main = main_in + "local/ready_features/"
avg_dir = local_feat_main + "averaged_data/"
avg_rcp45 = pd.read_csv(avg_dir + "averaged_data_rcp45.csv") 

all_data_usa.head(2)
avg_rcp45.head(2)

"""
Append

- First append the needed treated unit to all historical/control units so we can find nearest locations.
- Then take numeric part of the data to be able to operate on them.
    - normalize
    - find distances
- add the distances back to the data frame.
- sort the data frame according to distances.
"""

one_site = avg_rcp45.iloc[0, :].copy()
dt = all_data_usa.append(one_site)

# Take the numerical values so we can work with them, normalize them, compute KNNs
dt_numeric = dt.drop(['year', 'location', 'treatment', 'ClimateScenario'], axis=1).copy()
dt_numeric.tail(2)

# Normalize data
x = dt_numeric.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_dt = pd.DataFrame(x_scaled)

normalized_dt.tail(2)

"""
compute Euclidean distances
Subtract the last row from all other rows to compute distances.
"""

subtracted = normalized_dt - normalized_dt.iloc[-1, ]
subtracted.head(2)

distances = np.linalg.norm(subtracted, axis=1, keepdims=True)
dt['distances'] = distances
dt = dt.sort_values(by=['distances'])

out_name = one_site['location'] + '_' + str(one_site['year']) + ".csv"
dt.to_csv(out_dir + out_name, sep='\t')


