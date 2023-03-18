import numpy as np
import pandas as pd
import geopandas as gpd
import sys
from IPython.display import Image
from shapely.geometry import Point, Polygon
from math import factorial
import datetime
import time
import scipy

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb




def filter_out_unwanted(dt_df):
    unwanted_plants = ["Almond", "Apple", "Alfalfa/Grass Hay",
                       "Apricot", "Asparagus", "Berry, Unknown",  
                       "Blueberry", "Cherry", "Grape, Juice", 
                       "Grape, Table", "Grape, Unknown", 
                       "Grape, Wine", "Hops", "Mint", 
                       "Nectarine/Peach", "Orchard, Unknown", 
                       "Pear", "Plum", "Strawberry", "Walnut",
                       "Alfalfa Hay", "Alfalfa Seed", "Alfalfa Seed",
                       "Grass Hay", "Hay/Silage , Unknown", "Hay/Silage, Unknown",
                       "Pasture", "Timothy"]
    
    # filter unwanted plants
    dt_df = dt_df[~(dt_df['CropTyp'].isin(unwanted_plants))]
    
    # filter non-irrigated
    """
    # These two lines can replace the following two lines
    non_irrigations = ["Unknown", "None", "None/Rill", "None/Sprinkler", 
                       "None/Sprinkler/Wheel Line", 
                       "None/Wheel Line", "Drip/None", "Center Pivot/None"]
    
    dt_df = dt_df[~(dt_df['Irrigtn'].isin(non_irrigations))]
    """
    dt_df = dt_df[~dt_df['Irrigtn'].str.contains("None")]
    dt_df = dt_df[~dt_df['Irrigtn'].str.contains("Unknown")]
    
    return dt_df
    
def initial_clean(dt):
    # remove the useles system:index column
    if ("system:index" in list(dt.columns)):
        dt = dt.drop(columns=['system:index'])
    
    # Drop rows whith NA in NDVI column.
    dt = dt[dt['NDVI'].notna()]
    
    # rename the column .geo to "geo"
    dt = dt.rename(columns={".geo": "geo"})
    return (dt)

def order_by_doy(dt):
    return dt.sort_values(by='doy', axis=0, ascending=True)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    y_array = np.array(y)
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y_array[0] - np.abs( y_array[1:half_window+1][::-1] - y_array[0] )
    lastvals = y_array[-1] + np.abs(y_array[-half_window-1:-1][::-1] - y_array[-1])
    y_array = np.concatenate((firstvals, y_array, lastvals))
    return np.convolve( m[::-1], y_array, mode='valid')


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise (ValueError, 
                'Input vectors y_axis and x_axis must have same length')
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    https://github.com/mattijn/pynotebook/blob/16fe0f58624938b82d93cbd208b8cb871ab95ec1/
    ipynotebooks/Python2.7/.ipynb_checkpoints/PLOTS%20SIGNAL%20PROCESSING-P1%20and%20P2-checkpoint.ipynb
     
    also look at: https://gist.github.com/endolith/250860
    and 
    http://billauer.co.il/peakdet.html
    
    
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError ( "Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError ( "delta must be a positive number" )
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  # slows down this
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows down this
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]

def form_xs_ys_from_peakdetect(max_peak_list, doy_vect):
    dd = np.array(doy_vect)
    xs = np.zeros(len(max_peak_list))
    ys = np.zeros(len(max_peak_list))
    for ii in range(len(max_peak_list)):  
        xs[ii] = dd[int(max_peak_list[ii][0])]
        ys[ii] = max_peak_list[ii][1]
    return (xs, ys)

def keep_WSDA_columns(dt_dt):
    needed_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo']
    """
    # Using DataFrame.drop
    df.drop(df.columns[[1, 2]], axis=1, inplace=True)

    # drop by Name
    df1 = df1.drop(['B', 'C'], axis=1)
    """
    dt_dt = dt_dt[needed_columns]
    return dt_dt


def convert_TS_to_a_row(a_dt):
    a_dt = keep_WSDA_columns(a_dt)
    a_dt = a_dt.drop_duplicates()
    return(a_dt)
    



data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/01_NDVI_TS/Grant/"

file_names = ["Grant_2018_TS.csv"]
file_N = file_names[0]


a_df = pd.read_csv(data_dir + file_N)
a_df = initial_clean(a_df)
a_df.head(2)




# Count distict values, use nunique:
pprint (a_df['geo'].nunique())

# Count only non-null values, use count:
print (a_df['geo'].count())

# Count total values including null values, use size attribute:
print (a_df['geo'].size)







polygon_list = a_df['geo'].unique()






output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                  'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                  'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                  'peak_Doy', 'peak_value']

all_polygons_and_their_peaks = pd.DataFrame(data=None, 
                                            columns=output_columns)
for a_poly in polygon_list:
    a_field = a_df[a_df['geo']==a_poly]
    
    ### 
    ###  There is a chance that a polygon is repeated twice?
    ###
    
    X = a_field['doy']
    y = a_field['NDVI']
    freedom_df = 7
    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    
    # Generate spline basis with "freedom_df" degrees of freedom
    x_basis = cr(X, df=freedom_df, constraints='center')

    # Fit model to the data
    model = LinearRegression().fit(x_basis, y)

    # Get estimates
    y_hat = model.predict(x_basis)
    
    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    # peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
    # max_peaks = peaks_LWLS_1[0]
    # peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

    peaks_spline = peakdetect(y_hat, lookahead = 10, delta=0)
    max_peaks = peaks_spline[0]
    peaks_spline = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)
    
    DoYs_series = pd.Series(int(peaks_spline[0]))
    peaks_series = pd.Series(peaks_spline[1])
    
    peak_df = pd.DataFrame({ 
                       'peak_Doy': pd.Series(int(peaks_spline[0])), 
                       'peak_value': pd.Series(peaks_spline[1]) 
                      }) 

    
    WSDA_df = keep_WSDA_columns(a_field)
    WSDA_df = WSDA_df.drop_duplicates()
    
    WSDA_df = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index()
    # WSDA_df = pd.concat([WSDA_df, peak_df], axis=1, ignore_index=True)
    WSDA_df = WSDA_df.join(peak_df)
    
    """
    # first I decided to add all DoY and peaks in one row to avoid
    # multiple rows per (field, year)
    # However, in this way, each pair of (field, year)
    # can have different column sizes.
    # So, we cannot have one dataframe to include everything in it.
    # so, we will have to do dictionary to save out puts.
    # Lets just do replicates... easier to handle perhaps down the road.
    #
    DoY_colNames = [i + j for i, j in zip(\
                                          ["DoY_"]*(len(DoYs_series)+1), \
                                          [str(i) for i in range(1, len(DoYs_series)+1)] )] 
   
    peak_colNames = [i + j for i, j in zip(\
                                           ["peak_"]*(len(peaks_series)+1), \
                                           [str(i) for i in range(1, len(peaks_series)+1)] )]
    
    WSDA_df[DoY_colNames] = pd.DataFrame([DoYs_series], index=WSDA_df.index)
    WSDA_df[peak_colNames] = pd.DataFrame([peaks_series], index=WSDA_df.index)
    """





output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                  'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                  'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                  'peak_Doy', 'peak_value']

all_polygons_and_their_peaks = pd.DataFrame(data=None, 
                                            columns=output_columns)



a_field = a_df[a_df['geo']==polygon_list[0]]

### 
###  There is a chance that a polygon is repeated twice?
###

X = a_field['doy']
y = a_field['NDVI']
freedom_df = 7
#############################################
###
###             Smoothen
###
#############################################

# Generate spline basis with "freedom_df" degrees of freedom
x_basis = cr(X, df=freedom_df, constraints='center')

# Fit model to the data
model = LinearRegression().fit(x_basis, y)

# Get estimates
y_hat = model.predict(x_basis)

#############################################
###
###             find peaks
###
#############################################
# peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
# max_peaks = peaks_LWLS_1[0]
# peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

peaks_spline = peakdetect(y_hat, lookahead = 10, delta=0)
max_peaks = peaks_spline[0]
peaks_spline = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

DoYs_series = pd.Series(int(peaks_spline[0]))
peaks_series = pd.Series(peaks_spline[1])

peak_df = pd.DataFrame({ 
                   'peak_Doy': pd.Series(int(peaks_spline[0])), 
                   'peak_value': pd.Series(peaks_spline[1]) 
                  }) 

WSDA_df = keep_WSDA_columns(a_field)
WSDA_df = WSDA_df.drop_duplicates()





WSDA_df = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index(drop=True)
WSDA_df = WSDA_df.join(peak_df)


all_polygons_and_their_peaks.append(all_polygons_and_their_peaks, WSDA_df)


#########################
X = a_field['doy']
y = a_field['NDVI']
freedom_df = 7

# Generate spline basis with "freedom_df" degrees of freedom
x_basis = cr(X, df=freedom_df, constraints='center')

# Fit model to the data
model = LinearRegression().fit(x_basis, y)

# Get estimates
y_hat = model.predict(x_basis)



# find peaks
# peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
# max_peaks = peaks_LWLS_1[0]
# peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

peaks_spline = peakdetect(y_hat, lookahead = 10, delta=0)
max_peaks = peaks_spline[0]
peaks_spline = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)


plt.scatter(X, y, s=7);
plt.scatter(peaks_spline[0], peaks_spline[1], s=120, c='c', marker='*');
plt.plot(X, y_hat, 'r');
plt.title(f'Natural cubic spline with {freedom_df} degrees of freedom \n and peak point in red');


peaks_spline

WSDA_df = convert_TS_to_a_row(a_field)
WSDA_df
WSDA_df = WSDA_df.reset_index(drop=True)
WSDA_df

# DoYs_df = pd.Series(int(peaks_spline[0]), name='peak_Doy').to_frame()
# peaks_df = pd.Series(peaks_spline[1], name="peak_value").to_frame()

peak_df = pd.DataFrame({ 
                       'peak_Doy': pd.Series(int(peaks_spline[0])), 
                       'peak_value': pd.Series(peaks_spline[1]) 
                      }) 

peak_df = pd.DataFrame({ 
                       'peak_Doy': np.array([147, 148, 2]), 
                       'peak_value': np.array([10.2, 14.8, 33.4])
                      }) 

peak_df

AAA = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index(drop=True)
AAA

# pd.merge(WSDA_df, peak_df)
# pd.merge(left=WSDA_df, right=peak_df, how='outer', on='index')
pd.concat([AAA, peak_df], axis=1, ignore_index=True)

AAA.join(peak_df)


pd.concat([WSDA_df]*peak_df.shape[0]).reset_index(drop=True)

