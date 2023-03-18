rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
options(digit=9)
options(digits=9)
##########################################################################################
###                                                                                    ###
###                             Define Functions here                                  ###
###                                                                                    ###
##########################################################################################

produce_data_4_plots <- function(data, average_type="none"){
  needed_cols = c("Chill_season", "sum_J1", "sum_F1","year", "model", 
                  "scenario", "lat", "long", "climate_type")

  ################### CLEAN DATA
  data = subset(data, select=needed_cols)
  data = data %>% filter(year<=2005 | year>=2025)

  # time periods are
  time_periods = c("Historical","2025_2050", "2051_2075", "2076_2099")
    
  data$time_period = 0L
  data$time_period[data$year <= 2005] = time_periods[1]
  data$time_period[data$year >= 2025 & data$year <= 2050] = time_periods[2]
  data$time_period[data$year >= 2051 & data$year <= 2075] = time_periods[3]
  data$time_period[data$year >= 2076] = time_periods[4]
  data$time_period = factor(data$time_period, levels =time_periods, order=T)

  #################################################################
  #
  # Take Average over locations, or models, or none.
  #
  #################################################################
  if (average_type == "locations"){
    data <- data %>% 
            group_by(time_period, model, scenario, climate_type) %>%
            summarise_at(.funs = funs(averages = mean), vars(sum_J1:sum_F1)) %>%
            data.table()
    } else if (average_type == "models"){
    data <- data %>% 
            group_by(time_period, lat, long, scenario, climate_type) %>%
            summarise_at(.funs = funs(averages = mean), vars(sum_J1:sum_F1)) %>%
            data.table()
  }
  
  data_f <- data %>% filter(time_period != "Historical")
  data_h_rcp85 <- data %>% filter(time_period == "Historical")
  data_h_rcp45 <- data %>% filter(time_period == "Historical")

  data_h_rcp85$scenario = "RCP 8.5"
  data_h_rcp45$scenario = "RCP 4.5"

  # data$scenario[data$scenario=="historical"] = "Historical"
  data_f$scenario[data_f$scenario=="rcp45"] = "RCP 4.5"
  data_f$scenario[data_f$scenario=="rcp85"] = "RCP 8.5"
  
  data = rbind(data_f, data_h_rcp45, data_h_rcp85)
  rm(data_h_rcp45, data_h_rcp85, data_f)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##     Find the 90th percentile of the chill units                   ##
  ##     Grouped by location, model, time_period and rcp               ##
  ##     This could be used for box plots, later compute the mean.     ##
  ##     for maps                                                      ##
  ##                                                                   ##
  #######################################################################
  if (average_type == "locations"){
      quan_per_loc_period_model_jan <- data %>% 
                                       group_by(time_period, scenario, model, climate_type) %>%
                                       summarise(quan_90 = quantile(sum_J1_averages, probs = 0.1)) %>%
                                       data.table()
  
      quan_per_loc_period_model_feb <- data %>% 
                                       group_by(time_period, scenario, model, climate_type) %>%
                                       summarise(quan_90 = quantile(sum_F1_averages, probs = 0.1)) %>%
                                       data.table()
      
      # There will be no map for this case
      mean_quan_per_loc_period_model_jan = NA
      mean_quan_per_loc_period_model_feb = NA
      median_quan_per_loc_period_model_jan = NA
      median_quan_per_loc_period_model_feb = NA

      } else if (average_type == "models"){

      quan_per_loc_period_model_jan <- data %>% 
                                       group_by(time_period, lat, long, scenario, climate_type) %>%
                                       summarise(quan_90 = quantile(sum_J1_averages, probs = 0.1)) %>%
                                       data.table()
  
      quan_per_loc_period_model_feb <- data %>% 
                                       group_by(time_period, lat, long, scenario, climate_type) %>%
                                       summarise(quan_90 = quantile(sum_F1_averages, probs = 0.1)) %>%
                                       data.table()

      # There will be no map for this case
      mean_quan_per_loc_period_model_jan = NA
      mean_quan_per_loc_period_model_feb = NA
      median_quan_per_loc_period_model_jan = NA
      median_quan_per_loc_period_model_feb = NA

    } else if (average_type == "none") {
      quan_per_loc_period_model_jan <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, climate_type) %>%
                                   summarise(quan_90 = quantile(sum_J1, probs = 0.1)) %>%
                                   data.table()
  
      quan_per_loc_period_model_feb <- data %>% 
                                       group_by(time_period, lat, long, scenario, model, climate_type) %>%
                                       summarise(quan_90 = quantile(sum_F1, probs = 0.1)) %>%
                                       data.table()

      # it seems there is a library, perhaps tidyverse, that messes up
      # the above line, so the two variables above are 1-by-1. 
      # just close and re-open R Studio
        
      mean_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                            group_by(time_period, lat, long, scenario) %>%
                                            summarise(mean_over_model = mean(quan_90)) %>%
                                            data.table()

      mean_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                            group_by(time_period, lat, long, scenario) %>%
                                            summarise(mean_over_model = mean(quan_90)) %>%
                                            data.table()

      median_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                              group_by(time_period, lat, long, scenario) %>%
                                              summarise(mean_over_model = median(quan_90)) %>%
                                              data.table()

      median_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                              group_by(time_period, lat, long, scenario) %>%
                                              summarise(mean_over_model = median(quan_90)) %>%
                                              data.table()
    }

  return(list(quan_per_loc_period_model_jan, 
              mean_quan_per_loc_period_model_jan, 
              median_quan_per_loc_period_model_jan,
              quan_per_loc_period_model_feb, 
              mean_quan_per_loc_period_model_feb, 
              median_quan_per_loc_period_model_feb)
        )
}

#######################################################################
##                                                                   ##
##                             Driver                                ##
##                                                                   ##
#######################################################################
time_types = c("non_overlapping") # , "overlapping"
model_types = c("dynamic_model_stats") # , "utah_model_stats"
main_in = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling"
file_name = "summary_comp.rds"

avg_type = "models" # locations, models, none
time_type = time_types[1]
model_type = model_types[1]

for (time_type in time_types){
  for (model_type in model_types){
    in_dir = file.path(main_in, time_type, model_type, file_name)
    out_dir = file.path(main_in, time_type, model_type, "/")
    
    datas = data.table(readRDS(in_dir))
    information = produce_data_4_plots(datas, average_type = avg_type)

    safe_jan <- safe_box_plot(information[[1]], due="Jan.")
    safe_feb <- safe_box_plot(information[[4]], due="Feb.")
    
    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Jan_", avg_type, ".png")
    ggsave(output_name, safe_jan, path=out_dir, width=4, height=4, unit="in", dpi=400)

    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Feb_", avg_type, ".png")
    ggsave(output_name, safe_feb, path=out_dir, width=4, height=4, unit="in", dpi=400)
    
    # means over models
    # mean_map_jan = ensemble_map(data=information[[2]], color_col="mean_over_model", due="Jan.")
    # mean_map_feb = ensemble_map(data=information[[5]], color_col="mean_over_model", due="Feb.")

    # output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_jan.png") 
    # ggsave(output_name, mean_map_jan, path=out_dir, width=7, height=4.5, unit="in", dpi=400)

    # output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_feb.png") 
    # ggsave(output_name, mean_map_feb, path=out_dir, width=7, height=4.5, unit="in", dpi=400)  
  }
}

