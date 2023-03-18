
options(digits=9)
options(digits=9)

# source_path = "/home/hnoorazar/reading_binary/read_binary_core.R"
# source(source_path)
##################################################################

##################################################################
##################################################################
##################################################################

process_data_non_overlap_springer <- function(file, time_period) {
  # if (time_period == "2025_2050"){
  #   processed_data <- file %>%
  #                      filter(year >= 2025 & year <= 2050,
  #                             chill_season != "chill_2025-2026" &
  #                             chill_season != "chill_2050-2051"
  #                             )
  #   } else if (time_period == "2051_2075"){
  #    processed_data <- file %>%
  #                      filter(year > 2050 & year <= 2075,
  #                             chill_season != "chill_2050-2051" &
  #                             chill_season != "chill_2075-2076"
  #                             )
  #   } else if (time_period == "2076_2100") {
  #     processed_data <- file %>%
  #                       filter(year > 2075 & year <= 2099,
  #                              chill_season != "chill_2075-2076" &
  #                              chill_season != "chill_2099-2100"
  #                             )
   
  #   } else if (time_period == "2005_2024") {
  #    processed_data <- file %>%
  #                      filter(year >= 2005 & year <= 2024 ,
  #                             chill_season != "chill_2005-2006" &
  #                             chill_season != "chill_2024-2025"
  #                             )
  #  }
  processed_data <- file %>%
                    # Only want complete seasons of data
                    filter(chill_season != "chill_2005-2006" &
                           chill_season != "chill_2099-2100")
  processed_data <- na.omit(processed_data)

  processed_data <- threshold_func_Springer(processed_data, data_type="modeled")
  return(processed_data)
}


grab_coord_springer <- function(A){
  out_put <- A %>%
             transmute(lat = as.numeric(substr(x = .id, start = 19, stop = 26)),
                       long = as.numeric(substr(x = .id, start = 28, stop = 37)),
                       median_37half = median_37half,
                       median_55 = median_55)
    return (out_put)
}


get_medians_springer <- function(a_list){
  medians_data <- ldply(.data = a_list,
                        .fun = function(x) medians_springer(thresh_37half = x[, "thresh_37half"],
                                                            thresh_55 = x[, "thresh_55"]
                                                            ))
  return (medians_data)
}


# A function that pulls the median values we want
medians_springer <- function(thresh_37half, thresh_55) {
  c(median_37half = median(thresh_37half),
    median_55 = median(thresh_55)
    )
}


threshold_func_Springer <- function(file, data_type){
  file <- na.omit(file)
  if (data_type=="modeled"){
    data <- file %>%
           # Only want complete seasons of data
           filter(chill_season != "chill_1949-1950" & chill_season != "chill_2005-2006")
   } else {
    data <- file %>%
            # Only want complete seasons of data
            filter(chill_season != "chill_1978-1979" & chill_season != "chill_2015-2016")
  } 
  data <- data %>% 
          # Within a season
          group_by(chill_season) %>%
          # Mutate output is the row index of the first 
          # time where it meets threshold
          # within the group. (Index is the same as counting 
          # the start date as day = 1)

          mutate(thresh_37half = detect_index(.x = cume_portions,
                                              .f = chill_thresh,
                                               threshold = 37.5), # 37.5 comes from 75% of 50CP
                 thresh_55 =     detect_index(.x = cume_portions,
                                              .f = chill_thresh,
                                              threshold = 55)
                 ) %>% 
          summarise(sum = sum(daily_portions),
                    thresh_37half = unique(thresh_37half), # retain the thresholds
                    thresh_55 = unique(thresh_55)
                    ) %>%
          data.frame() # to allow for ldply() later
  return(data)
}

Accum_GDD_DoYDetector_basedOnCaneldarYear <- function(data, data_type, cut_off = 55){
  ###
  ###   This function is modification of the function heat_trigger_DoYDetector(.)
  ###
  ###   This function finds the DoY at which 55 vertical GDD is accumulated.
  ###   March 2, 2021. HN
  ###

  data <- na.omit(data)
  # if (data_type=="modeled"){
  #   data <- data %>%
  #          # Only want complete seasons of data
  #          filter(chill_season != "chill_1949-1950" & chill_season != "chill_2005-2006")
  #  } else {
  #   data <- data %>%
  #           # Only want complete seasons of data
  #           filter(chill_season != "chill_1978-1979" & chill_season != "chill_2015-2016")
  # } 
  data <- data %>% 
          # Within a season
          group_by(year, location, model, emission) %>%
          # Mutate output is the row index of the first 
          # time where it meets threshold
          # within the group. (Index is the same as counting 
          # the start date as day = 1)
          mutate(heatTriggerThresh = detect_index(.x = vert_Cum_dd_F,
                                                  .f = chill_thresh,
                                                  threshold = cut_off)
                ) %>% 
          summarise(heatTriggerThresh = unique(heatTriggerThresh), # retain the thresholds
                    ) %>%
          data.frame() # to allow for ldply() later
  return(data)
}
##################################################################
##################################################################
##################################################################
grab_coord_heatTrigger <- function(A){
  out_put <- A %>%
             transmute(lat = as.numeric(substr(x = .id, start = 19, stop = 26)),
                       long = as.numeric(substr(x = .id, start = 28, stop = 37)),
                       median_heatTriggerThresh = median_heatTriggerThresh)
  return (out_put)
}

get_medians_heatTrigger <- function(a_list){
  medians_data <- ldply(.data = a_list,
                        .fun = function(x) medians_heatTrigger(heatTriggerThresh = x[, "heatTriggerThresh"]))
  return (medians_data)
}

medians_heatTrigger <- function(heatTriggerThresh) {
  c(median_heatTriggerThresh = median(heatTriggerThresh))
}


heat_trigger_DoYDetector <- function(file, data_type, min_CP = 73.3, percent_4_heat = 0.75){
  ###
  ###   This function is modification of the function threshold_func(.)
  ###
  ###   Lets say a given cultivar needs at least X CP.
  ###   This function finds the DoY at which 75% of the X is met.
  ###   Reason: percent_4_heat = 75% is what is needed for the heat accumulation to start!!!
  ###   
  ###   The default value of min_CP=73.3 is for CrippsPink that is
  ###   Obtained from: 
  ###      Chilling requirements of apple cultivars grown in mild Australian winter conditions
  ###
  ###

  cut_off <- percent_4_heat * min_CP


  file <- na.omit(file)
  if (data_type=="modeled"){
    data <- file %>%
           # Only want complete seasons of data
           filter(chill_season != "chill_1949-1950" & chill_season != "chill_2005-2006")
   } else {
    data <- file %>%
            # Only want complete seasons of data
            filter(chill_season != "chill_1978-1979" & chill_season != "chill_2015-2016")
  } 
  data <- data %>% 
          # Within a season
          group_by(chill_season) %>%
          # Mutate output is the row index of the first 
          # time where it meets threshold
          # within the group. (Index is the same as counting 
          # the start date as day = 1)
          mutate(heatTriggerThresh = detect_index(.x = cume_portions,
                                                  .f = chill_thresh,
                                                  threshold = cut_off)
                ) %>% 
          summarise(sum = sum(daily_portions),
                    heatTriggerThresh = unique(heatTriggerThresh), # retain the thresholds
                    ) %>%
          data.frame() # to allow for ldply() later
  return(data)
}

#############################################

add_time_periods_model <- function(dt){
  time_periods <- c("1950-2005", 
                    "2006-2025", "2026-2050", 
                    "2051-2075", "2076-2099")
  dt$time_period <- 0L
  dt$time_period[dt$year <= 2005] <- time_periods[1]
  dt$time_period[dt$year >= 2006 & dt$year <= 2025] <- time_periods[2]
  dt$time_period[dt$year >= 2026 & dt$year <= 2050] <- time_periods[3]
  dt$time_period[dt$year >= 2051 & dt$year <= 2075] <- time_periods[4]
  dt$time_period[dt$year >= 2076] <- time_periods[5]
  return(dt)
}

add_time_periods_observed <- function(dt){
  dt$time_period <- c("1979-2015")
  return(dt)
}

put_time_period <- function(data_tb, observed){
  if (observed==TRUE){
     data_tb$time_period <- "1979-2016"
     } else {
      data_tb <- data_tb %>%
                 mutate(time_period = case_when(year %in% c(1950:2005) ~ "1950-2005",
                                                year %in% c(2006:2025) ~ "2006-2025",
                                                year %in% c(2026:2050) ~ "2026-2050",
                                                year %in% c(2051:2075) ~ "2051-2075",
                                                year %in% c(2076:2099) ~ "2076-2099")
                        )
  }
  return(data_tb)
}

thresh_median_4_frost_bloom_thresh_plot <- function(data_tb, colname){
  colnames(data_tb)[colnames(data_tb)==colname] <- "thresh"
  data_tb <- data_tb[, .(thresh_median_DoY = median(thresh)), 
                    by = c("city", "year", "chill_season", 
                           "emission", "time_period")]
  return(data_tb)
}

remove_F0 <- function(data_tb){
  if ("year" %in% colnames(data_tb)){
    data_tb <- data_tb %>% 
               filter(!(year %in% c(2015:2025))) %>%
               data.table() 
  } else if ("time_period" %in% colnames(data_tb)){
    data_tb <- data_tb %>% 
               filter(time_period != "2015-2025") %>%
               data.table() 
  }
  return(data_tb)
}

remove_modeled_historical_add_time_period <- function(data_tb){
  # here we try to remove modeled historical from a data table
  # that does not have time_period in it.
  # and all RCPs along with models are binded together 
  # 
  setnames(data_tb, old=c("scenario"), new=c("emission"))
  
  data_tb_observed <- data_tb %>% 
                      filter(model %in% c("observed", "Observed")) %>% 
                      data.table()

  data_tb_observed_45 <- data_tb_observed
  data_tb_observed_85 <- data_tb_observed

  data_tb_observed_45$emission <- "RCP 4.5"
  data_tb_observed_85$emission <- "RCP 8.5"
  data_tb_observed <- rbind(data_tb_observed_45, data_tb_observed_85)
  data_tb_observed$time_period <- "observed"

  data_tb_modeled <- data_tb %>% 
                     filter(!(model %in% c("observed", "Observed"))) %>% 
                     data.table()

  # we can go up to 2015 to make the plots continuous.
  data_tb_modeled <- data_tb_modeled %>% 
                     filter(!(year %in% c(1950:2025))) %>% 
                     data.table()

  data_tb_modeled$time_period <- "future"

  data_tb <- rbind(data_tb_observed, data_tb_modeled)
  data_tb$emission[data_tb$emission=="rcp45"] <- "RCP 4.5"
  data_tb$emission[data_tb$emission=="rcp85"] <- "RCP 8.5"
  return(data_tb)
}

#
#  Bloom DoY that bloom is completed by X% (e.g. 50%)
#
median_bool_DoY <- function(data_dt){
  data_dt$dayofyear <- as.double(data_dt$dayofyear)

  data_dt = data_dt[, .(median_DoY = median(dayofyear)), 
                      by = c("time_period", "emission", "year","city")]
  return(data_dt)
}

bloom_completed_x_percent <- function(data_dt, colname, thresh){
  # get the data that are more than the threshold
  # then, get the minimums per group!

  data_dt <- data_dt %>% filter(get(colname) >= thresh) %>% data.table()
  data_dt <- data_dt[ , .SD[which.min(get(colname))], 
                         by = c("time_period", "year", "emission", "city", "model")]
  return(data_dt)
}


##################################################################

form_chill_season_day_of_year_observed <- function(data){
  ################################
  ####
  #### Remove out-of-boundary years
  ####
  #
  # Toss unwanted data
  #
  data <- data.table(data)
  data <- data %>% filter(month %in% c(9, 10, 11, 12, 1, 2))
  data <- data %>% filter(!(year == 1979 & month %in% c(1, 2)))
  data <- data %>% filter(!(year == 2016 & month %in% c(9, 10, 11, 12)))
  
  # Set January and Feb. to 13th and 14th month of the year
  # so, we can sort the months, and compute day of year, beginning
  # Sept.
  data$month[data$month == 1] = 13
  data$month[data$month == 2] = 14

  ######## Reduce the year of the Jan and Feb so they are in the right chill season
  data$year[data$month == 13] = data$year[data$month == 13] - 1
  data$year[data$month == 14] = data$year[data$month == 14] - 1

  # keycol <- c("year", "month", "day")
  # setorderv(data, keycol)
  data <- data.table(data)
  data$chill_dayofyear <- 1 # dummy
  data[, chill_dayofyear := cumsum(chill_dayofyear), by=list(year)]
  data <- data.table(data)
  return(data)
}

form_chill_season_day_of_year_modeled <- function(data){
  ################################
  ####
  #### Remove out-of-boundary years
  ####
  #
  # Toss unwanted data
  #
  data <- data.table(data)
  data <- data %>% filter(month %in% c(9, 10, 11, 12, 1, 2))
  data <- data %>% filter(!(year == 1950 & month %in% c(1, 2)))
  data <- data %>% filter(!(year == 2005 & month %in% c(9, 10, 11, 12)))
  data <- data %>% filter(!(year == 2006 & month %in% c(1, 2)))
  data <- data %>% filter(!(year == 2099 & month %in% c(9, 10, 11, 12)))

  # Set January and Feb. to 13th and 14th month of the year
  # so, we can sort the months, and compute day of year, beginning
  # Sept.
  data$month[data$month == 1] = 13
  data$month[data$month == 2] = 14

  ######## Reduce the year of the Jan and Feb so they are in the right chill season
  data$year[data$month == 13] = data$year[data$month == 13] - 1
  data$year[data$month == 14] = data$year[data$month == 14] - 1

  # keycol <- c("year", "month", "day")
  # setorderv(data, keycol)
  data <- data.table(data)
  data$chill_dayofyear <- 1 # dummy
  data[, chill_dayofyear := cumsum(chill_dayofyear), by=list(year)]
  data <- data.table(data)
  return(data)
}

kth_smallest_in_group <- function(dt, target_column, k){
  result <- dt %>% 
            group_by(location, year, model) %>%
            arrange(get(target_column)) %>%
            slice(k) %>%
            data.table()
  return(result)
}

pick_single_cities_by_location <- function(dt, city_info){
  if (!("location" %in% colnames(city_info))){
    city_info$location <- paste0(city_info$lat, "_", city_info$long)
    city_info <- within(city_info, remove(lat, long))
  }

  if (!("location" %in% colnames(dt))){
    dt$location <- paste0(dt$lat, "_", dt$long)
    dt <- within(dt, remove(lat, long))
  }
  dt <- dt %>% 
        filter(location %in% city_info$location) %>%
        data.table()
  dt <- merge(dt, city_info)
  return(dt)
}

# remove_montana_add_warm_cold <- function(data_dt, LocationGroups_NoMontana){
#   data_dt <- remove_montana(data_dt, LocationGroups_NoMontana)
#   data_dt <- left_join(x=data_dt, y=LocationGroups_NoMontana)
#   return(data_dt)
# }

remove_montana <- function(data_dt, LocationGroups_NoMontana){
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$lat, "_", data_dt$long)
  }
  data_dt <- data_dt %>% filter(location %in% LocationGroups_NoMontana$location)
  return(data_dt)
}

###################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!

count_years_threshs_met_all_locations <- function(dataT, due){
  h_year_count <- length(unique(dataT[dataT$time_period =="Historical",]$chill_season))
  f1_year_count <- length(unique(dataT[dataT$time_period== "2025_2050",]$chill_season))
  f2_year_count <- length(unique(dataT[dataT$time_period== "2051_2075",]$chill_season))
  f3_year_count <- length(unique(dataT[dataT$time_period== "2076_2099",]$chill_season))
  if (due == "Jan"){
     col_name = "sum_J1"
     } else if(due == "Feb"){
      col_name = "sum_F1"
     } else if(due =="Mar"){
      col_name = "sum_M1"
     } else if(due =="Apr"){
      col_name = "sum_A1"
     } else if (due=="mid_Apr"){
     col_name = "sum_mid_Apr"
     } else if (due=="May"){
     col_name = "sum_May1"
     } else if (due=="mid_May"){
     col_name = "sum_mid_May"
    }

  bks = c(-300, seq(20, 75, 5), 300)
  if (!("location" %in% colnames(dataT))){
    dataT$location = paste0(dataT$lat, "_", dataT$long)
    dataT <- within(dataT, remove("lat", "long"))
  }

  dataT_hist <- dataT %>% filter(scenario == "Historical")
  dataT_45 <- dataT %>% filter(scenario == "RCP 4.5")
  dataT_85 <- dataT %>% filter(scenario == "RCP 8.5")

  dataT_hist <- droplevels(dataT_hist)
  dataT_45 <- droplevels(dataT_45)
  dataT_85 <- droplevels(dataT_85)

  result_85 <- dataT_85 %>%
               mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
               tidyr::complete(time_period, thresh_range, model, location) %>%
               group_by(time_period, thresh_range, model, location) %>%
               summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
               data.table()

  result_45 <- dataT_45 %>%
               mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
               tidyr::complete(time_period, thresh_range, model, location) %>%
               group_by(time_period, thresh_range, model, location) %>%
               summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
               data.table()

  result_H <- dataT_hist %>%
              mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
              tidyr::complete(time_period, thresh_range, model, location) %>%
              group_by(time_period, thresh_range, model, location) %>%
              summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
              data.table()

  # we do this, so historical appears in both plots
  result_85 <- rbind(result_85, result_H)
  result_45 <- rbind(result_45, result_H)

  result_85$scenario <- "RCP 8.5"; result_45$scenario <- "RCP 4.5"
  result <- rbind(result_45, result_85)

  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  result$time_period = factor(result$time_period, levels=time_periods, order=T)
  
  result$thresh_range <- factor(result$thresh_range, order=T)
  result$thresh_range <- fct_rev(result$thresh_range)
  result <- result[order(thresh_range), ]

  result <- result %>% 
            group_by(time_period, model, scenario, location) %>% 
            mutate(n_years_passed = cumsum(no_years)) %>% 
            data.table()
  
  result_hist <- result %>% filter(time_period == "Historical") %>% data.table()
  result_50 <- result %>% filter(time_period == "2025_2050") %>% data.table()
  result_75 <- result %>% filter(time_period == "2051_2075") %>% data.table()
  result_99 <- result %>% filter(time_period == "2076_2099") %>% data.table()
  
  result_hist$frac_passed = result_hist$n_years_passed / h_year_count
  result_50$frac_passed = result_50$n_years_passed / f1_year_count
  result_75$frac_passed = result_75$n_years_passed / f2_year_count
  result_99$frac_passed = result_99$n_years_passed / f3_year_count

  result <- rbind(result_hist, result_50, result_75, result_99)
  result <- na.omit(result)
  return(result)
}

count_years_threshs_met_limit_location <- function(dataT, due){
  h_year_count <- length(unique(dataT[dataT$time_period =="Historical",]$chill_season))
  f1_year_count <- length(unique(dataT[dataT$time_period== "2025_2050",]$chill_season))
  f2_year_count <- length(unique(dataT[dataT$time_period== "2051_2075",]$chill_season))
  f3_year_count <- length(unique(dataT[dataT$time_period== "2076_2099",]$chill_season))
  if (due == "Jan"){
     col_name = "sum_J1"
     } else if (due == "Feb"){
      col_name = "sum_F1"
     } else if(due =="Mar"){
      col_name = "sum_M1"
     } else if(due =="Apr"){
      col_name = "sum_A1"
     } else if (due=="mid_Apr"){
      col_name = "sum_mid_Apr"
     } else if (due=="May"){
     col_name = "sum_May1"
     } else if (due=="mid_May"){
     col_name = "sum_mid_May"
    }

  bks = c(-300, seq(20, 75, 5), 300)

  # df_help[1, 2:8] = table(cut(data_hist_rich$Temp, breaks = iof_breaks))
  dataT_hist <- dataT %>% filter(scenario == "Historical")
  dataT_45 <- dataT %>% filter(scenario == "RCP 4.5")
  dataT_85 <- dataT %>% filter(scenario == "RCP 8.5")

  dataT_hist <- droplevels(dataT_hist)
  dataT_45 <- droplevels(dataT_45)
  dataT_85 <- droplevels(dataT_85)

  result_85 <- dataT_85 %>%
               mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
               complete(time_period, thresh_range, model, city) %>%
               group_by(time_period, thresh_range, model, city) %>%
               summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
               data.table()

  result_45 <- dataT_45 %>%
               mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
               complete(time_period, thresh_range, model, city) %>%
               group_by(time_period, thresh_range, model, city) %>%
               summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
               data.table()

  result_H <- dataT_hist %>%
              mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
              complete(time_period, thresh_range, model, city) %>%
              group_by(time_period, thresh_range, model, city) %>%
              summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
              data.table()

  # we do this, so historical appears in both plots
  result_85 <- rbind(result_85, result_H)
  result_45 <- rbind(result_45, result_H)

  result_85$scenario <- "RCP 8.5"
  result_45$scenario <- "RCP 4.5"

  result <- rbind(result_45, result_85)

  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  result$time_period = factor(result$time_period, levels=time_periods, order=T)
  
  result$thresh_range <- factor(result$thresh_range, order=T)
  result$thresh_range <- fct_rev(result$thresh_range)
  result <- result[order(thresh_range), ]

  result <- result %>% 
            group_by(time_period, model, scenario, city) %>% 
            mutate(n_years_passed = cumsum(no_years)) %>% 
            data.table()
  
  result_hist <- result %>% filter(time_period == "Historical") %>% data.table()
  result_50 <- result %>% filter(time_period == "2025_2050") %>% data.table()
  result_75 <- result %>% filter(time_period == "2051_2075") %>% data.table()
  result_99 <- result %>% filter(time_period == "2076_2099") %>% data.table()
  
  result_hist$frac_passed = result_hist$n_years_passed / h_year_count
  result_50$frac_passed = result_50$n_years_passed / f1_year_count
  result_75$frac_passed = result_75$n_years_passed / f2_year_count
  result_99$frac_passed = result_99$n_years_passed / f3_year_count

  result <- rbind(result_hist, result_50, result_75, result_99)
  result <- na.omit(result)
  return(result)
}

########################################
######
######    Folder 02_ stuff
######
########################################

grab_coord <- function(A){
  out_put <- A %>%
             transmute(lat = as.numeric(substr(x = .id, start = 19, stop = 26)),
                       long = as.numeric(substr(x = .id, start = 28, stop = 37)),
                       median_20 = median_20,
                       median_25 = median_25,
                       median_30 = median_30,
                       median_35 = median_35,
                       median_40 = median_40,
                       median_45 = median_45,
                       median_50 = median_50,
                       median_55 = median_55,
                       median_60 = median_60,
                       median_65 = median_65,
                       median_70 = median_70,
                       median_75 = median_75,
                       median_J1 = median_J1,
                       median_F1 = median_F1,
                       median_M1 = median_M1,
                       median_A1 = median_A1)
    return (out_put)
}

get_medians <- function(a_list){
  medians_data <- ldply(.data = a_list,
                        .fun = function(x) medians(thresh_20 = x[, "thresh_20"],
                                                   thresh_25 = x[, "thresh_25"],
                                                   thresh_30 = x[, "thresh_30"],
                                                   thresh_35 = x[, "thresh_35"],
                                                   thresh_40 = x[, "thresh_40"],
                                                   thresh_45 = x[, "thresh_45"],
                                                   thresh_50 = x[, "thresh_50"],
                                                   thresh_55 = x[, "thresh_55"],
                                                   thresh_60 = x[, "thresh_60"],
                                                   thresh_65 = x[, "thresh_65"],
                                                   thresh_70 = x[, "thresh_70"],
                                                   thresh_75 = x[, "thresh_75"],
                                                   sum_J1 = x[, "sum_J1"],
                                                   sum_F1 = x[, "sum_F1"],
                                                   sum_M1 = x[, "sum_M1"],
                                                   sum_A1 = x[, "sum_A1"],
                                                   sum_mid_Apr = x[, "sum_mid_Apr"],
                                                   sum_May1 = x[, "sum_May1"],
                                                   sum_mid_May = x[, "sum_mid_May"]
                                                   ))
  return (medians_data)
}

# A function that pulls the median values we want
medians <- function(thresh_20, thresh_25, thresh_30,
                    thresh_35, thresh_40, thresh_45,
                    thresh_50, thresh_55, thresh_60,
                    thresh_65, thresh_70, thresh_75, 
                    sum_J1, sum_F1, sum_M1, sum_A1, sum_mid_Apr,
                    sum_May1, sum_mid_May) {
  c(median_20 = median(thresh_20),
    median_25 = median(thresh_25),
    median_30 = median(thresh_30),
    median_35 = median(thresh_35),
    median_40 = median(thresh_40),
    median_45 = median(thresh_45),
    median_50 = median(thresh_50),
    median_55 = median(thresh_55),
    median_60 = median(thresh_60),
    median_65 = median(thresh_65),
    median_70 = median(thresh_70),
    median_75 = median(thresh_75),
    median_J1 = median(sum_J1),
    median_F1 = median(sum_F1),
    median_M1 = median(sum_M1),
    median_A1 = median(sum_A1),
    median_mid_Apr = median(sum_mid_Apr),
    median_May1 = median(sum_May1),
    median_sum_mid_May = median(sum_mid_May)
    )
}
##################################################
#####                                        #####
#####    this is the overlapping one         #####
#####    the non-overlap is below            #####
#####                                        #####
##################################################
# process_data <- function(file, time_period) { # this is for overlapping data, which we never will use!
#   if (time_period=="2040"){
#     processed_data <- file %>%
#                       filter(year >= 2025 & year <= 2055,
#                              chill_season != "chill_2025-2026" &
#                              chill_season != "chill_2055-2056")
#   } else if (time_period=="2060"){
#     processed_data <- file %>%
#                       filter(year > 2045 & year <= 2075,
#                              chill_season != "chill_2045-2046" &
#                              chill_season != "chill_2075-2076")
#   } else if (time_period=="2080") {
#      processed_data <- file %>%
#                        filter(year > 2065 & year <= 2095,
#                               chill_season != "chill_2065-2066" &
#                               chill_season != "chill_2095-2096")
#   }
#   processed_data <- threshold_func(processed_data, data_type="modeled")
#   return(processed_data)
# }

##################################################
#####                                        #####
#####    this is the non-overlapping one     #####
#####                                        #####
##################################################
process_data_non_overlap <- function(file, time_period) {
  # if (time_period == "2025_2050"){
  #   processed_data <- file %>%
  #                      filter(year >= 2025 & year <= 2050,
  #                             chill_season != "chill_2025-2026" &
  #                             chill_season != "chill_2050-2051"
  #                             )
  #   } else if (time_period == "2051_2075"){
  #    processed_data <- file %>%
  #                      filter(year > 2050 & year <= 2075,
  #                             chill_season != "chill_2050-2051" &
  #                             chill_season != "chill_2075-2076"
  #                             )
  #   } else if (time_period == "2076_2100") {
  #     processed_data <- file %>%
  #                       filter(year > 2075 & year <= 2099,
  #                              chill_season != "chill_2075-2076" &
  #                              chill_season != "chill_2099-2100"
  #                             )
   
  #   } else if (time_period == "2005_2024") {
  #    processed_data <- file %>%
  #                      filter(year >= 2005 & year <= 2024 ,
  #                             chill_season != "chill_2005-2006" &
  #                             chill_season != "chill_2024-2025"
  #                             )
  #  }
  processed_data <- file %>%
                    # Only want complete seasons of data
                    filter(chill_season != "chill_2005-2006" &
                           chill_season != "chill_2099-2100")
  processed_data <- na.omit(processed_data)

  processed_data <- threshold_func(processed_data, data_type="modeled")
  return(processed_data)
}

threshold_func <- function(file, data_type){
  file <- na.omit(file)
  if (data_type=="modeled"){
    data <- file %>%
           # Only want complete seasons of data
           filter(chill_season != "chill_1949-1950" & chill_season != "chill_2005-2006")
   } else {
    data <- file %>%
            # Only want complete seasons of data
            filter(chill_season != "chill_1978-1979" & chill_season != "chill_2015-2016")
  } 
  data <- data %>% 
          # Within a season
          group_by(chill_season) %>%
          # Mutate output is the row index of the first 
          # time where it meets threshold
          # within the group. (Index is the same as counting 
          # the start date as day = 1)
          mutate(thresh_20 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 20),
                 thresh_25 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 25),
                 thresh_30 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 30),
                 thresh_35 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 35),
                 thresh_40 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 40),
                 thresh_45 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 45),
                 thresh_50 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 50),
                 thresh_55 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 55),
                 thresh_60 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 60),
                 thresh_65 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 65),
                 thresh_70 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 70),
                 thresh_75 = detect_index(.x = cume_portions,
                                          .f = chill_thresh,
                                          threshold = 75),
                 # Below we set the row where, ex. month = 1 (jan) and day = 1 to be
                 # equal to the value of cume_portions. This creates one non-NA row
                 # for the whole column. Then in summarise we remove all NAs, which
                 # collapses the column to a single value -- perfect for a one-line
                 # summary per season.
                 sum_J1 = case_when(month == 1 & day == 1 ~ cume_portions),
                 sum_F1 = case_when(month == 2 & day == 1 ~ cume_portions),
                 sum_M1 = case_when(month == 3 & day == 1 ~ cume_portions),
                 sum_A1 = case_when(month == 4 & day == 1 ~ cume_portions),
                 sum_mid_Apr = case_when(month == 4 & day == 15 ~ cume_portions),
                 sum_May1 = case_when(month == 5 & day == 1 ~ cume_portions),
                 sum_mid_May = case_when(month == 5 & day == 15 ~ cume_portions)
                 ) %>% 
          summarise(sum = sum(daily_portions),
                    thresh_20 = unique(thresh_20), # retain the thresholds
                    thresh_25 = unique(thresh_25),
                    thresh_30 = unique(thresh_30),
                    thresh_35 = unique(thresh_35),
                    thresh_40 = unique(thresh_40),
                    thresh_45 = unique(thresh_45),
                    thresh_50 = unique(thresh_50),
                    thresh_55 = unique(thresh_55),
                    thresh_60 = unique(thresh_60),
                    thresh_65 = unique(thresh_65),
                    thresh_70 = unique(thresh_70),
                    thresh_75 = unique(thresh_75),
                    sum_J1 = na.omit(sum_J1),
                    sum_F1 = na.omit(sum_F1),
                    sum_M1 = na.omit(sum_M1),
                    sum_A1 = na.omit(sum_A1),
                    sum_mid_Apr = na.omit(sum_mid_Apr),
                    sum_May1 = na.omit(sum_May1),
                    sum_mid_May = na.omit(sum_mid_May)
                    ) %>%
          data.frame() # to allow for ldply() later
  return(data)
}

# A function to check against a threshold
chill_thresh <- function(x, threshold) {x >= threshold}

###################################################################

put_chill_season <- function(data_tb, chill_start = "sept"){

  if (chill_start == "sept"){
    #########################
    #
    # Chill season start at Sep.
    #
    #########################
    data_tb <- data_tb %>%
               mutate(chill_season = case_when(
                      # If Jan:Aug then part of chill season of prev year - current year
                      month %in% c(1:8) ~ paste0("chill_", (year - 1), "-", year),
                      # If Sept:Dec then part of chill season of current year - next year
                      month %in% c(9:12) ~ paste0("chill_", year, "-", (year + 1))
                      ))
    } else if (chill_start == "mid_sept"){
      #########################
      #
      # Chill season start at Mid Sep.
      #
      #########################
      data_tb <- data_tb %>%
                 mutate(chill_season = case_when(
                        # If Jan:Sept_15th then part of chill season of prev year - current year                
                        month %in% c(1:8) ~ paste0("chill_", (year - 1), "-", year),
                        ((month %in% c(9)) & (day <= 15)) ~ paste0("chill_", (year - 1), "-", year),

                        # If Sept_16th:Dec then part of chill season of current year - next year
                        ((month %in% c(9)) & (day >= 16)) ~ paste0("chill_", year, "-", (year + 1)),
                        (month %in% c(10:12)) ~ paste0("chill_", year, "-", (year + 1))
                        ))
      } else if (chill_start == "oct"){
      #########################
      #
      # Chill season start at Oct
      #
      #########################
      data_tb <- data_tb %>%
                 mutate(chill_season = case_when(
                        # If Jan:Sept then part of chill season of prev year - current year
                        month %in% c(1:9) ~ paste0("chill_", (year - 1), "-", year),
                        # If Oct:Dec then part of chill season of current year - next year
                        month %in% c(10:12) ~ paste0("chill_", year, "-", (year + 1))
                        ))
      } else if (chill_start == "mid_oct"){
      #########################
      #
      # Chill season start at Mid Oct
      #
      #########################
      data_tb <- data_tb %>%
                 mutate(chill_season = case_when(
                        # If Jan:oct_15th then part of chill season of prev year - current year                
                        month %in% c(1:9) ~ paste0("chill_", (year - 1), "-", year),
                        ((month %in% c(10)) & (day <= 15)) ~ paste0("chill_", (year - 1), "-", year),
                        # If oct_16th:Dec then part of chill season of current year - next year
                        ((month %in% c(10)) & (day >= 16)) ~ paste0("chill_", year, "-", (year + 1)),
                        month %in% c(11:12) ~ paste0("chill_", year, "-", (year + 1))
                        ))
    } else if (chill_start == "nov"){
      #########################
      #
      # Chill season start at Nov
      #
      #########################
      data_tb <- data_tb %>%
                 mutate(chill_season = case_when(
                        # If Jan:Nov then part of chill season of prev year - current year
                        month %in% c(1:10) ~ paste0("chill_", (year - 1), "-", year),
                        # If Nov:Dec then part of chill season of current year - next year
                        month %in% c(11:12) ~ paste0("chill_", year, "-", (year + 1))
                        ))
    } else if (chill_start == "mid_nov"){
      #########################
      #
      # Chill season start at Mid Nov
      #
      #########################
      data_tb <- data_tb %>%
                 mutate(chill_season = case_when(
                        # If Jan:Nov_15th then part of chill season of prev year - current year                
                        month %in% c(1:10) ~ paste0("chill_", (year - 1), "-", year),
                        ((month %in% c(11)) & (day <= 15)) ~ paste0("chill_", (year - 1), "-", year),
                        # If Nov_16th:Dec then part of chill season of current year - next year
                        ((month %in% c(11)) & (day >= 16)) ~ paste0("chill_", year, "-", (year + 1)),
                        month %in% c(12) ~ paste0("chill_", year, "-", (year + 1))
                        ))
  }
  return(data_tb)
}


projec_diff_from_hist <- function(dt_dt, diff_from="Observed"){
  #
  # diff_from in {"1979-2016", "1950-2005"}
  #
  if (diff_from=="Observed"){
      
      dt_dt <- dt_dt %>% 
           filter(time_period != "modeled hist")%>% 
           data.table()

      dt_dt <- dt_dt %>% 
           filter(time_period != "1950-2005")%>% 
           data.table()
     } else {
     dt_dt <- dt_dt %>% 
           filter(time_period != "Observed")%>% 
           data.table()
  }

  # we have to have unique historical to be able
  # to subtract it from future stuff
  # suppressWarnings({ dt_dt_hist <- dt_dt %>% 
  #                                  filter(time_period == diff_from)%>% 
  #                                  select(-c("emission")) %>%
  #                                  unique() %>% 
  #                                  data.table()
  #                 })

  # dt_dt_hist$emission = "hist"
  # dt_dt_hist$model = "hist"
  # dt_dt <- dt_dt %>% 
  #          filter(time_period != diff_from)%>% 
  #          data.table()
  
  # dt_dt <- rbind(dt_dt, dt_dt_hist)

  # melt to get differences
  dt_dt <- melt(dt_dt, id = c("location", "model", "time_period", "emission"))

  dt_dt <- within(dt_dt, remove("variable"))

  setnames(dt_dt, old = c("value"), new = c("CP_median_A1"))

  diffs <- dt_dt %>%
           group_by(location) %>%
           mutate(CP_diff = CP_median_A1 - CP_median_A1[time_period == diff_from])%>%
           data.table()

  # remove the historical data itself for which diffs. are zeros
  diffs <- diffs %>% filter(model != "observed")
  diffs <- diffs %>% filter(time_period != "1950-2005")
  diffs <- diffs %>% filter(time_period != "1979-2015")
  

  # to do percentages
  #**** The following 4 lines can be replaced by the 5th one ****
  # histor <- dt_dt %>% filter(model == "hist") %>% data.table()
  # histor <- subset(histor, select=c("location", "time_interval", "storm_value"))
  # setnames(histor, old="storm_value", new="hist_storm_val")
  # diffs <- merge(diffs, histor, by= c("location", "time_interval"), all.x=T)
  
  diffs$hist_CP <- diffs$CP_median_A1 +  diffs$CP_diff

  diffs$perc_diff <- (diffs$CP_diff * 100) / (diffs$hist_CP)
  return(diffs)
}


produce_relaxedSC_4_plots_w_noly_observed <- function(data){
  needed_cols = c("chill_season", "sum_A1", "year", "model", 
                  "emission", "lat", "long", "location", "time_period")

  ################### CLEAN DATA
  data = subset(data, select=needed_cols)
  # pick up future data
  # data = data %>% filter(emission != "historical")
  data <- organize_time_period_two_hist_w_only_observed(data)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##   Find the 90th percentile of the chill units                     ##
  ##   Grouped by location, model, time_period and rcp                 ##
  ##   This could be used for box plots, later compute the mean.       ##
  ##   for maps                                                        ##
  ##                                                                   ##
  #######################################################################

  quan_per_loc_period_model_apr <- data %>% 
                                   group_by(time_period, emission, model, location) %>%
                                   summarise(quan_80 = quantile(sum_A1, probs = 0.05)) %>%
                                   data.table()

  # it seems there is a library, perhaps tidyverse, that messes up
  # the above line, so the two variables above are 1-by-1. 
  # just close and re-open R Studio
  ########################################################################
  #######                                                          #######
  #######                     Means                                #######
  #######                                                          #######
  ########################################################################
  
  mean_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                        group_by(time_period, location, emission) %>%
                                        summarise(mean_over_model = mean(quan_80)) %>%
                                        data.table()
  ########################################################################
  #######                                                          #######
  #######                     Medians                              #######
  #######                                                          #######
  ########################################################################
  
  median_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                          group_by(time_period, location, emission) %>%
                                          summarise(median_over_model = median(quan_80)) %>%
                                          data.table()
  
  # quan_per_loc_period_model_jan$time_period = factor(quan_per_loc_period_model_jan$time_period, order=T)
  # mean_quan_per_loc_period_model_jan$time_period = factor(mean_quan_per_loc_period_model_jan$time_period, order=T)
  # median_quan_per_loc_period_model_jan$time_period = factor(median_quan_per_loc_period_model_jan$time_period, order=T)

  # quan_per_loc_period_model_feb$time_period_feb = factor(quan_per_loc_period_model_feb$time_period, order=T)
  # mean_quan_per_loc_period_model_feb$time_period = factor(mean_quan_per_loc_period_model_feb$time_period, order=T)
  # median_quan_per_loc_period_model_feb$time_period= factor(median_quan_per_loc_period_model_feb$time_period, order=T)

  return(list(quan_per_loc_period_model_apr,
              mean_quan_per_loc_period_model_apr,
              median_quan_per_loc_period_model_apr
              )
        )
}
