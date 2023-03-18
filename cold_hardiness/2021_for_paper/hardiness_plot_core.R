library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)

Markus_Figure_3 <- function(temp_data, 
                            CH_data,
                            temp_time_peirod = "observed", # observed, modeled_hist, 2025-2050, 2051-2075, 2076-2100
                            CH_time_period = "2025-2050",  # observed, modeled_hist, 2025-2050, 2051-2075, 2076-2100
                            grape_varieties = c("Merlot", "Riesling")){
    ##
    ##
    ## This function is written to replicate the Figure 3 in Markus' paper:
    ## Modeling Dormant Bud Cold Hardiness and Budbreak in Twenty-Three Vitis Genotypes Reveals Variation by Region of Origin
    ##
    ## input: temp_data that includes observed data, modeled historical, and future temperatures
    ##
    ##        CH_data   that includes CH data from observed temps, CH data from modeled historical temps , 
    ##                  and CH data from future temperatures
    ##
    ##        future_time_period: The time period we want to be plotted from future.
    ##
    ## output: the figure
    ##
    
}

