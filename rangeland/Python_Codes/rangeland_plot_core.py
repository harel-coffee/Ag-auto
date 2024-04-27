import pandas as pd
import numpy as np

import sys, scipy, scipy.signal

import datetime
from datetime import date, timedelta
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import seaborn as sb


def makeColorColumn(gdf, variable, vmin, vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGnBu)
    gdf["value_determined_color"] = gdf[variable].apply(
        lambda x: mcolors.to_hex(mapper.to_rgba(x))
    )
    return gdf
