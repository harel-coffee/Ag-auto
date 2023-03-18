
#
# Define directories
#
data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/01_NDVI_TS/Grant/"
#
# Define some parameters
#
file_names = ["Grant_2018_TS.csv"]

#
# Read Data
#
file_N = file_names[0]
grant_2018 = pd.read_csv(data_dir + file_N)

#
# clean
#
grant_2018 = initial_clean(grant_2018)

#
# List of unique polygons
#
polygon_list = grant_2018['geo'].unique()

#
# Pick one of the polygons and test the smoothing methods
#
grant_2018_first_field = grant_2018[grant_2018['geo']==polygon_list[1]]
grant_2018_first_field = order_by_doy(grant_2018_first_field)
print(grant_2018_first_field.shape)
print(type(grant_2018_first_field))

# #
# # 1. Weighted Least Square
# #

# X = grant_2018_first_field['doy']
# y = grant_2018_first_field['NDVI']

# res_wls = sm.WLS(endog=y, exog=X, weights=y**(2)).fit()
# res_wls.params


# prstd, iv_l, iv_u = wls_prediction_std(res_wls)
# prstd, iv_l, iv_u = wls_prediction_std(res_wls)


# ##
# ## Plot
# ##
# fig, ax = plt.subplots(figsize=(8,6));
# ax.plot(X, y, 'o', label="Data");

# ax.plot(X, res_wls.fittedvalues, 'g--.', label="WLS fitted values")
# # ax.plot(X, iv_u, 'g--', label="WLS")
# # ax.plot(X, iv_l, 'g--')
# ax.legend(loc="best");

"""
Locally weighted regression
   - The [documentation page](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html) 
   from stats model package
   
In the following the parameters are:

   - `frac`:  Between 0 and 1. The fraction of the data used when estimating each y-value.
   - `itint`: The number of residual-based reweightings to perform.
"""


lowess = sm.nonparametric.lowess
LWLS_1 = lowess(endog=y, exog=X, frac= 1./3, it=0)
LWLS_2 = lowess(endog=y, exog=X, frac= 1./3)
LWLS_3 = lowess(endog=y, exog=X, frac= 1./3, it=5)
# predict_3 = lowess(endog=y, exog=X)

##
## Plot
##
fig, ax = plt.subplots(figsize=(8,6));
ax.plot(X, y, 'o', label="Data");

ax.plot(X, LWLS_1[:, 1], 'g--.', label="predict_1")
ax.plot(X, LWLS_2[:, 1], 'r--.', label="predict_2")
ax.plot(X, LWLS_3[:, 1], 'b--.', label="predict_3")
ax.legend(loc="best");


##
##  Savitzky Golay filtering
##
smoothed_by_Savitzky = savitzky_golay(y, window_size=5, order=1)


##
## Plot
##
fig, ax = plt.subplots(figsize=(8,6));
ax.plot(X, y, 'o', label="Data");
ax.plot(X, LWLS_1[:, 1], 'g--.', label="predict_1 by LWLS")
ax.plot(X, LWLS_2[:, 1], 'r--.', label="predict_2 by LWLS")
ax.plot(X, LWLS_3[:, 1], 'b--.', label="predict_3 by LWLS")
ax.plot(X, smoothed_by_Savitzky, 'k--.', label="smoothed_by_Savitzky")
ax.legend(loc="best");

"""
## Gaussian Filtering/convolution 
[Documentation page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
"""
gaussian_smoothed = scipy.ndimage.gaussian_filter(input=y, sigma=2.5, order=0)

##
## Plot
##

fig, ax = plt.subplots(figsize=(8,6));
ax.plot(X, y, 'o', label="Data");

ax.plot(X, LWLS_1[:, 1], 'g--.', label="predict_1 by LWLS")
ax.plot(X, LWLS_3[:, 1], 'b--.', label="predict_3 by LWLS")
ax.plot(X, smoothed_by_Savitzky, 'k--.', label="Savitzky")
ax.plot(X, gaussian_smoothed, 'r--.', label="Gaussian")
ax.legend(loc="best");

##
## Spline Smooothing
##
freedom_df = 7

# Generate spline basis with 10 degrees of freedom
x_basis = cr(X, df=freedom_df, constraints='center')

# Fit model to the data
model = LinearRegression().fit(x_basis, y)

# Get estimates
y_hat = model.predict(x_basis)

##
## Plot
##
plt.scatter(X, y, s=7)
plt.plot(X, y_hat, 'r', label="smoothing spline result");
plt.title(f'Natural cubic spline with {freedom_df} degrees of freedom')



# find peaks
peaks = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
max_peaks = peaks[0]
peaks = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

##
## Plot peaks
##
plt.scatter(X, y, s=7);
plt.scatter(peaks[0], peaks[1], s=8);
plt.plot(X, y_hat, 'r');
plt.title(f'Natural cubic spline with {freedom_df} degrees of freedom \n and peak point in red');

