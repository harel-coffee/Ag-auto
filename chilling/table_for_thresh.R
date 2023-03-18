
rm(list=ls())

library(data.table)
library(ggpubr)
library(plyr)
library(tidyverse)
library(ggplot2)
########################################################################################
#######                                                                          #######
#######                Add time periods with and without overlap                 #######
#######                                                                          #######
########################################################################################

add_time_period <- function(data_in, time_period_type="non_overlapping"){
    if (time_period_type == "overlapping"){
        all_data = data.table()
        data_hist = filter(data_in, year<=2005)
        data_2040 = filter(data_in, year>2025 & year<=2055)
        data_2060 = filter(data_in, year>2045 & year<=2075)
        data_2080 = filter(data_in, year>2065 & year<=2085)
        rm(data_in)

        data_hist$ClimateGroup = "Historical"
        data_2040$ClimateGroup = "2040's"
        data_2060$ClimateGroup = "2060's"
        data_2080$ClimateGroup = "2080's"

        all_data = rbind(data_hist, data_2040, data_2060, data_2080)
        
        time_periods = c("Historical", "2040's", "2060's", "2080's")

        all_data$ClimateGroup <- factor(all_data$ClimateGroup)
        all_data$ClimateGroup <- ordered(all_data$ClimateGroup, levels = time_periods)
        all_data <- within(all_data, remove(year))
        return(all_data)

    } else if (time_period_type == "non_overlapping"){
        all_data = data.table()
        data_hist = filter(data_in, year<=2005)
        data_2025_2050 = filter(data_in, year>2025 & year<=2050)
        data_2051_2075 = filter(data_in, year>2050 & year<=2075)
        data_2076_2100 = filter(data_in, year>2075 & year<=2100)
        rm(data_in)

        data_hist$ClimateGroup = "Historical"
        data_2025_2050$ClimateGroup = "2025_2050"
        data_2051_2075$ClimateGroup = "2051_2075"
        data_2076_2100$ClimateGroup = "2076_2100"
        all_data = rbind(data_hist, data_2025_2050, data_2051_2075, data_2076_2100)

        time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2100")
        all_data$ClimateGroup <- factor(all_data$ClimateGroup)
        all_data$ClimateGroup <- ordered(all_data$ClimateGroup, levels = time_periods)
        all_data <- within(all_data, remove(year))
        return(all_data)
    }
}

clean_data <- function(data){
    data = within(data, remove(sum, sum_J1, sum_F1, sum_M1, sum_A1, 
                               lat, long, Chill_season))
    data = filter(data, model != "observed" )

    data = within(data, remove(model))
    return(data)
}

########################################################################################

model_dir_postfix = "_model_stats/"

time_period_types = c("non_overlapping") # , "overlapping"
model_types = c("dynamic") # , "utah"
main_data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"

for (time_period_type in time_period_types){
  for (model_type in model_types){
    data_dir = paste0(main_data_dir, time_period_type, "/", model_type, model_dir_postfix)
    data <- data.table(readRDS(paste0(data_dir, "summary_comp.rds")))

    # remove extra columns
    data = clean_data(data)
    data <- add_time_period(data_in=data, time_period_type=time_period_type)

    #######
    #######     Compute Medians
    #######
    # The two followings are the same!
    data_medians <- data %>% 
                    group_by(climate_type, scenario, ClimateGroup) %>% # climate_type is warm or cold which we do not have any more
                    summarise_at(.funs = funs(med = median), vars(thresh_20:thresh_75)) %>%
                    data.table()

    data_90th <- data %>% 
                 group_by(climate_type, scenario, ClimateGroup) %>%
                 summarise_all(list(Q90 = quantile), probs = 0.9) %>%
                 data.table()
    #######
    #######     Compute InterQuantile Ranges
    #######
    data_IQR <- data %>% group_by(climate_type, scenario, ClimateGroup) %>%
                         summarise_at(.funs = funs(IntQntRng = IQR), vars(thresh_20:thresh_75))%>%
                  data.table()
    #######
    #######     merge medians and IQRs
    #######
    # data_merged = merge(data_medians, data_IQR, by=c("climate_type", "scenario", "ClimateGroup"))

    #######
    #######     Writing Time
    #######
    write_dir_main = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
    write_dir = paste0(write_dir_main, time_period_type, "/", model_type, "_model_stats/tables/")
    
    write_dir <- "/Users/hn/Desktop/"
    write.table(data_medians, file = paste0(write_dir, model_type,"_medians.csv"), 
                row.names = FALSE, 
                col.names = TRUE, 
                sep = ",")

    write.table(data_IQR, file = paste0(write_dir, model_type, "_IQR.csv"), 
                row.names = FALSE, 
                col.names = TRUE, 
                sep = ",")

    write.table(data_90th, file = paste0(write_dir, model_type, "_quan_90.csv"), 
                row.names = FALSE, 
                col.names = TRUE, 
                sep = ",")

    # write.table(data_merged, file = paste0(write_dir, model_type, "_medians_IQR.csv"),
    #             row.names = FALSE, 
    #             col.names = TRUE, 
    #             sep = ",")
    }
}
