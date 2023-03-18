options(digits=9)
options(digits=9)

####################
#
# Lots of stuff here are inefficient.
# I did do cumulative sum of different things
# and then had to do extra step of filtering the
# last rows corresponding to a particular column or time period
# 
# I did that because there are a lot of unpredictable
# things that may come up!!!!!!
#
# Instead of cum. sum, we could just sum stuff in a given
# column for a given time period or else, like so:
# data <- data %>% 
#              group_by(location, year, season, model, emission) %>% 
#              summarise(seasonal_cum_precip = sum(monthly_cum_precip)) %>% 
#              data.table()

# However, the advantage of the first period is that it will
# retain more information, that for debugging purposes can be used.
#
####################

##########################################################################
# source_path = "/home/hnoorazar/reading_binary/read_binary_core.R"      #
# source(source_path)                                                    #
##########################################################################
get_ridof_canada <- function(dt_dt){
  vect <- c("49.03125_-122.46875", "49.03125_-122.53125",
            "49.03125_-122.59375", "49.03125_-122.65625",
            "49.03125_-122.71875")
  dt_dt <- dt_dt %>% filter(!(location %in% vect))
  return(dt_dt)
}

remove_current_timeP <- function(data_tb){
 data_tb %>% filter(time_period != "2006-2025") %>% data.table()
}

remove_modeled_hist <- function(data_tb){
  data_tb %>% filter(time_period != "1950-2005")%>%data.table()
}
remove_observed <- function(data_tb){
  data_tb %>% filter(time_period != "1979-2016")%>%data.table()
}

convert_5_numeric_clusts_to_alphabet <- function(data_tb){
  data_tb <- data_tb %>%
             mutate(alph_clust = case_when(cluster == 1 ~ "Western coastal",
                                           cluster == 2 ~ "Cascade foothills",
                                           cluster == 3 ~ "Northwest Cascades",
                                           cluster == 4 ~ "Northcentral Cascades",
                                           cluster == 5 ~ "Northeast Cascades")) %>% 
             data.table()
  data_tb <- within(data_tb, remove(cluster))
  setnames(data_tb, old=c("alph_clust"), new=c("cluster"))
  cluster_levels <- c("Western coastal", "Cascade foothills", 
                      "Northwest Cascades", "Northcentral Cascades", 
                      "Northeast Cascades")
  data_tb$cluster <- factor(data_tb$cluster, levels=cluster_levels)

  return(data_tb)
}

update_clusters <- function(data_tb, new_clusters){
  if (length(unique(data_tb$cluster)) == 4){
    # some of the locations are absent in the Min's data
    data_tb <- data_tb %>% filter(location %in% new_clusters$location) %>% data.table()
    data_tb <- within(data_tb, remove(cluster))
    data_tb <- merge(data_tb, new_clusters, by="location", all.x=TRUE)
  }
  data_tb <- na.omit(data_tb) # some locations were missing in Min's data

  data_tb <- data_tb[, cluster:=as.character(cluster)]
  cluster_levels <- c("1", "2", "3", "4", "5")
  data_tb$cluster <- factor(data_tb$cluster, levels=cluster_levels)
  return(data_tb)
}

# find 25th and 75th percentiles per groups
find_quantiles <- function(data_table, tgt_col, time_type){

  if (time_type=="annual"){
     v <- annual_quantiles(data_table, tgt_col)
     } else if(time_type=="wtr_yr"){
      v <- annual_quantiles(data_table, tgt_col)
     }
      else if(time_type=="seasonal"){
      v <- seasonal_quantiles(data_table, tgt_col)
     } else if(time_type=="monthly"){
      v <- monthly_quantiles(data_table, tgt_col)
  }
  return (v)
}

monthly_quantiles <- function(data, tgt_col){
  quan_25 <- data %>% 
             group_by(time_period, emission, cluster, month) %>% 
             summarise(quan_25 = quantile(get(tgt_col), probs = 0.25)) %>% 
             data.table()

  quan_75 <- data %>% 
             group_by(time_period, emission, cluster, month) %>% 
             summarise(quan_75 = quantile(get(tgt_col), probs = 0.75)) %>% 
             data.table()

  both_quans <- merge(quan_25, quan_75, 
                      by=c("time_period", "emission", "cluster", "month"),
                      all.x=TRUE)

  both_quans$IQR = both_quans$quan_75 - both_quans$quan_25
  
  both_quans$quan_25 <- both_quans$quan_25 - (1.52 * both_quans$IQR)
  both_quans$quan_75 <- both_quans$quan_75 + (1.52 * both_quans$IQR)

  v <- c(min(both_quans$quan_25), max(both_quans$quan_75))
  return(v)
}

seasonal_quantiles <- function(data, tgt_col){
  quan_25 <- data %>% 
             group_by(time_period, emission, cluster, season) %>% 
             summarise(quan_25 = quantile(get(tgt_col), probs = 0.25)) %>% 
             data.table()

  quan_75 <- data %>% 
             group_by(time_period, emission, cluster, season) %>% 
             summarise(quan_75 = quantile(get(tgt_col), probs = 0.75)) %>% 
             data.table()

  both_quans <- merge(quan_25, quan_75, 
                      by=c("time_period", "emission", "cluster", "season"),
                      all.x=TRUE)

  both_quans$IQR = both_quans$quan_75 - both_quans$quan_25
  
  both_quans$quan_25 <- both_quans$quan_25 - (1.52 * both_quans$IQR)
  both_quans$quan_75 <- both_quans$quan_75 + (1.52 * both_quans$IQR)

  v <- c(min(both_quans$quan_25), max(both_quans$quan_75))
  return(v)
}

annual_quantiles <- function(data_tbl, tgt_col){
  quan_25 <- data_tbl %>% 
             group_by(time_period, emission, cluster) %>% 
             summarise(quan_25 = quantile(get(tgt_col), probs = 0.25)) %>% 
             data.table()

  quan_75 <- data_tbl %>% 
             group_by(time_period, emission, cluster) %>% 
             summarise(quan_75 = quantile(get(tgt_col), probs = 0.75)) %>% 
             data.table()

  both_quans <- merge(quan_25, quan_75, 
                      by=c("time_period", "emission", "cluster"),
                      all.x=TRUE)
  both_quans$IQR = both_quans$quan_75 - both_quans$quan_25
  
  both_quans$quan_25 <- both_quans$quan_25 - (1.52 * both_quans$IQR)
  both_quans$quan_75 <- both_quans$quan_75 + (1.52 * both_quans$IQR)

  v <- c(min(both_quans$quan_25), max(both_quans$quan_75))
  return(v)
}

storm_25_quantiles <- function(data_tbl, tgt_col="twenty_five_years"){
  quan_25 <- data_tbl %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise(quan_25 = quantile(get(tgt_col), probs = 0.25)) %>% 
             data.table()

  quan_75 <- data_tbl %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise(quan_75 = quantile(get(tgt_col), probs = 0.75)) %>% 
             data.table()

  both_quans <- merge(quan_25, quan_75, 
                      by=c("return_period", "emission", "cluster"),
                      all.x=TRUE)
  both_quans$IQR = both_quans$quan_75 - both_quans$quan_25
  
  both_quans$quan_25 <- both_quans$quan_25 - (1.52 * both_quans$IQR)
  both_quans$quan_75 <- both_quans$quan_75 + (1.52 * both_quans$IQR)

  v <- c(min(both_quans$quan_25), max(both_quans$quan_75))
  return(v)
}
##########################################################################
############
############ seasonal
############
seasonal_cum <- function(data_tb, material){
  # input : material \in {precip, rain, snow, runbase}
  data_tb <- put_season(data_tb)

  if (material == "precip"){
    return(seasonal_cum_precip(data_tb))
    } else if (material == "rain"){
    return(seasonal_cum_rain(data_tb))
    } else if (material == "snow"){
      return(seasonal_cum_snow(data_tb))
    } else if (material == "runbase"){
      return(seasonal_cum_runoff(data_tb))
  }
}

seasonal_cum_precip <- function(data_tb){
  
  data_tb <- data_tb %>% 
             group_by(location, year, season, 
                      model, emission, time_period,
                      cluster) %>% 
             summarise(seasonal_cum_precip = sum(monthly_cum_precip)) %>% 
             data.table()
  return(data_tb)
}

seasonal_cum_rain <- function(data_tb){
  data_tb <- data_tb %>% 
             group_by(location, year, season, 
                      model, emission, time_period,
                      cluster) %>% 
             summarise(seasonal_cum_rain = sum(monthly_cum_rain)) %>% 
             data.table()
  return(data_tb)
}

seasonal_cum_snow <- function(data_tb){
  data_tb <- data_tb %>% 
             group_by(location, year, season, 
                      model, emission, time_period,
                      cluster) %>% 
             summarise(seasonal_cum_snow = sum(monthly_cum_snow)) %>% 
             data.table()
  return(data_tb)
}

seasonal_cum_runoff <- function(data_tb){
  data_tb <- data_tb %>% 
             group_by(location, year, season, 
                      model, emission, time_period,
                      cluster) %>% 
             summarise(seasonal_cum_runbase = sum(monthly_cum_runbase)) %>% 
             data.table()
  return(data_tb)
}

put_season <- function(data){
  data <- data %>%
          mutate(season = case_when(month %in% c(9, 10, 11) ~ "fall",
                                    month %in% c(12, 1, 2) ~ "winter",
                                    month %in% c(3, 4, 5) ~ "spring",
                                    month %in% c(6, 7, 8) ~ "summer")
                ) %>% data.table()

  season_order <- c(12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
  # data$month <- factor(data$month, levels=season_order)
  return(data)
}
##########################################################################
find_10_90_quantiles <- function(dt_tba){
  # dt_tba is of type melted.
  return(quantile(dt_tba$value, probs = c(0.1, 0.9)))
}

find_5_95_quantiles <- function(dt_tba){
  # dt_tba is of type melted.
  return(quantile(dt_tba$value, probs = c(0.05, 0.95)))
}
########################################################################
monthly_cum_rain <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output:                                                    #
  #                                                            #
  #                                                            #
  ##############################################################
  data_tb$rain <- data_tb$precip * data_tb$rain_portion
  data_tb$snow <- data_tb$precip * (1 - data_tb$rain_portion)

  data_tb <- data_tb %>%
             group_by(location, year, month, model, emission) %>%
             mutate(monthly_cum_precip = cumsum(precip)) %>%
             data.table()

   data_tb <- data_tb %>%
            group_by(location, year, month, model, emission) %>%
            mutate(monthly_cum_rain = cumsum(rain)) %>%
            data.table()

  data_tb <- data_tb %>%
            group_by(location, year, month, model, emission) %>%
            mutate(monthly_cum_snow = cumsum(snow)) %>%
            data.table()
  return (data_tb)
}

wtr_yr_cum_rain <- function(data_tb){
  #
  # input: data_tb has to have the water_year column in it
  # output: cumulative precip in each water_year
  #
  data_tb$rain <- data_tb$precip * data_tb$rain_portion
  data_tb$snow <- data_tb$precip * (1 - data_tb$rain_portion)
  
  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(annual_cum_precip = cumsum(precip)) %>%
             data.table()

  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(annual_cum_rain = cumsum(rain)) %>%
             data.table()

  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(annual_cum_snow = cumsum(snow)) %>%
             data.table()

  return (data_tb)
}

chunky_cum_rain <- function(data_tb, start_month, end_month){
  #
  # first, put water_calendar so proper months are grouped together
  # second, toss out unwanted months.
  # third, find, cumu. over a wtr_year
  #
  data_tb <- create_wtr_calendar(data_tb, wtr_yr_start = start_month)
  data_tb <- data_tb %>%
             filter(!(month %in% ((end_month+1):(start_month-1))))%>%
             data.table()

  data_tb$rain <- data_tb$precip * data_tb$rain_portion
  data_tb$snow <- data_tb$precip * (1 - data_tb$rain_portion)

  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(chunk_cum_precip = cumsum(precip)) %>%
             data.table()

  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(chunk_cum_rain = cumsum(rain)) %>%
             data.table()
  data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(chunk_cum_snow = cumsum(snow)) %>%
             data.table()

 return(data_tb)
}

annual_cum_rain <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output:                                                    #
  #                                                            #
  #                                                            #
  ##############################################################
  data_tb$rain <- data_tb$precip * data_tb$rain_portion
  data_tb$snow <- data_tb$precip * (1 - data_tb$rain_portion)
  
  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_precip = cumsum(precip)) %>%
             data.table()
  
  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_rain = cumsum(rain)) %>%
             data.table()

  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_snow = cumsum(snow)) %>%
             data.table()

  return (data_tb)
}

rain_portion <- function(dt_dt){
  # dt_dt$rain_portion <- apply(dt_dt[, "tmean"], MARGIN=1, FUN=rain_p)
  # dt_dt$rain_portion <- unlist(lapply(data$tmean, FUN=rain_portion))

  dt_dt <- dt_dt %>%
           mutate(rain_portion = case_when(tmean <= 0.6 ~ 0,
                                           (tmean < 3.6 & tmean> 0.6) ~ ((tmean/3) - 0.2),
                                            tmean >= 3.6 ~ 1)) %>%
           data.table()
  return(dt_dt)
}

# rain_p <- function(tmean, Tt=2, Tr=13){
#   num <- tmean - Tt
#   denom <- 1.4 * Tr
#   frac <- num / denom

#   if (tmean <= Tt){
#       prain <- (5 * (frac^3)) + (6.76 * (frac^2)) + (3.19 * frac) + 0.5
#     } else {
#       prain <- (5 * (frac^3)) - (6.76 * (frac^2)) + (3.19 * frac) + 0.5
#   }
#   if (prain > 1){
#     prain <- 1
#   }
#   if (prain<0){
#     prain <- 0
#   }
#   return(prain)
# }
########################################################################
########################################################################
#####
#####         Diff from modeled historical
#####         Do we have to do extra stuff for this?
#####
###########################################################################
storm_diff_obs_or_modeled <- function(dt_dt, diff_from){
  if (diff_from=="1979-2016"){
      storm_diffs <- storm_diff(dt_dt, diff_from="1979-2016")
     } else {
      # HERE we have: diff_from=="1950-2005"
      storm_diffs <- data.table()
      
      # The followins is not necessary, it is done in compute... function
      dt_dt <- dt_dt %>% 
               filter(return_period != "2006-2025" & return_period != "1979-2016") %>% 
               data.table()
      all_mods <- unique(dt_dt$model)
      for (mod in all_mods){
         curr_dt <- dt_dt %>% filter(model == mod) %>% data.table()
         curr_dt <- storm_diff(curr_dt, diff_from="1950-2005")
         storm_diffs <- rbind(storm_diffs, curr_dt)
      }
  }
  return(storm_diffs)
}

storm_diff <- function(dt_dt, diff_from="1979-2016"){
  #
  # diff_from in {"1979-2016", "1950-2005"}
  #
  if (diff_from=="1979-2016"){
      toss_rp <- "1950-2005"
     } else {
     toss_rp <- "1979-2016"
  }
  dt_dt <- dt_dt %>% 
           filter(return_period != toss_rp & 
                  return_period != "2006-2025")%>% 
           data.table()

  # we have to have unique historical to be able
  # to subtract it from future stuff
  suppressWarnings({ dt_dt_hist <- dt_dt %>% 
                     filter(return_period == diff_from)%>% 
                     select(-c("emission")) %>%
                     unique() %>% 
                     data.table()})
  dt_dt_hist$emission = "hist"
  dt_dt_hist$model = "hist"

  dt_dt <- dt_dt %>% 
           filter(return_period != diff_from)%>% 
           data.table()
  
  dt_dt <- rbind(dt_dt, dt_dt_hist)

  # melt to get differences
  dt_dt <- melt(dt_dt, id = c("location", "model",
                              "return_period", "emission",
                              "cluster"))

  setnames(dt_dt, old = c("variable", "value"), 
                  new = c("time_interval", "storm_value"))

  diffs <- dt_dt %>%
           group_by(location, time_interval) %>%
           mutate(storm_diff = storm_value - storm_value[return_period == diff_from])%>%
           data.table()

  # remove the historical data itself for which diffs. are zeros
  diffs <- diffs %>% filter(model != "hist")

  # to do percentages
  #**** The following 4 lines can be replaced by the 5th one ****
  # histor <- dt_dt %>% filter(model == "hist") %>% data.table()
  # histor <- subset(histor, select=c("location", "time_interval", "storm_value"))
  # setnames(histor, old="storm_value", new="hist_storm_val")
  # diffs <- merge(diffs, histor, by= c("location", "time_interval"), all.x=T)
  
  diffs$hist_storm_val <- diffs$storm_value +  diffs$storm_diff

  diffs$perc_diff <- (diffs$storm_diff * 100) / (diffs$hist_storm_val)
  return(diffs)
}

####################################################################################
median_diff_obs_or_modeled_seasonal <- function(dt, tgt_col, diff_from){
  if (diff_from=="1979-2016"){
    median_diffs <- compute_median_diff_seasonal(dt, tgt_col, diff_from="1979-2016")
    
    } else {
      # HERE we have: diff_from=="1950-2005"
      median_diffs <- data.table()
      
      # The followings is not necessary, it is done in compute... function
      dt <- dt %>% 
            filter(time_period != "2006-2025" & 
                   time_period != "1979-2016") %>% 
            data.table()
      all_mods <- unique(dt$model)
      for (mod in all_mods){
        curr_dt <- dt %>% filter(model == mod) %>% data.table()
        curr_dt <- compute_median_diff_seasonal(curr_dt, tgt_col, diff_from="1950-2005")
        median_diffs <- rbind(median_diffs, curr_dt)
      }
  }
  return(median_diffs)
}

compute_median_diff_seasonal <- function(dt, tgt_col, diff_from="1979-2016"){
  #
  # Clean unwanted data
  #
  if (diff_from == "1979-2016"){
     unwanted_period <- "1950-2005"
     } else {
     unwanted_period <- "1979-2016"
  }

  dt <- dt %>% 
        filter(time_period != unwanted_period & 
               time_period != "2006-2025") %>% 
        data.table()

  if ("evap" %in% colnames(dt)){
    dt <- within(dt, remove(evap, runoff, base_flow, run_p_base))
  }
  if ("wtr_yr" %in% colnames(dt)) {
    dt <- within(dt, remove(wtr_yr))
  }
  suppressWarnings({dt <- within(dt, remove(year, month, day, precip))})

  ################################################
  #
  # We need to do the following lines in order
  # to make the subtraction within groups work.
  #
  dt_hist <- dt %>% 
             filter(time_period == diff_from) %>%
             select(-c("emission")) %>%
             unique()%>%
             data.table()

  dt_hist$emission <- "hist"
  dt <- dt %>% filter(time_period != diff_from) %>% data.table()
  dt <- rbind(dt, dt_hist); rm(dt_hist)

  med_per_loc_mod_TP_em <- dt[, .(tgt_col=median(get(tgt_col))), 
                               by = c("location", "time_period", 
                                      "emission", "model", "season",
                                      "cluster")]
  setnames(med_per_loc_mod_TP_em, old="tgt_col", new="medians")

  median_diffs <- med_per_loc_mod_TP_em %>%
                  group_by(location, season) %>%
                  mutate(diff = medians - medians[time_period == diff_from])%>%
                  filter(time_period != diff_from)%>%
                  data.table()

  median_diffs$hist_median <- median_diffs$medians - median_diffs$diff
  median_diffs$perc_diff <- (median_diffs$diff * 100) / (median_diffs$hist_median)
  return(median_diffs)
}
####################################################################################
median_of_diff_of_medians_month <- function(dt){
  #
  # tgt_col \in (diff, perc_diff)
  #
  diffs <- dt %>%
           group_by(location, time_period, emission, month) %>%
           transmute(med_of_diffs_of_meds = median(diff))%>%
           unique() %>%
           data.table()

  perc_diffs <- dt %>%
                group_by(location, time_period, emission, month) %>%
                transmute(perc_med_of_diffs_of_meds = median(perc_diff))%>%
                unique() %>%
                data.table()

  diffs <- merge(diffs, perc_diffs, 
                 by=c("location", 'emission', "time_period", "month"))
  cols <- c("med_of_diffs_of_meds", "perc_med_of_diffs_of_meds")
  diffs[,(cols) := round(.SD, 1), .SDcols=cols]
  return(diffs)
}

median_diff_obs_or_modeled_month <- function(dt, tgt_col, diff_from){
  if (diff_from=="1979-2016"){
    median_diffs <- compute_median_diff_month(dt, tgt_col, diff_from="1979-2016")
    
    } else {
      # HERE we have: diff_from=="1950-2005"
      median_diffs <- data.table()
      
      # The followings is not necessary, it is done in compute... function
      dt <- dt %>% 
            filter(time_period != "2006-2025" & 
                   time_period != "1979-2016") %>% 
            data.table()
      all_mods <- unique(dt$model)
      for (mod in all_mods){
        curr_dt <- dt %>% filter(model == mod) %>% data.table()
        curr_dt <- compute_median_diff_month(curr_dt, tgt_col, diff_from="1950-2005")
        median_diffs <- rbind(median_diffs, curr_dt)
      }
  }
  return(median_diffs)
}

compute_median_diff_month <- function(dt, tgt_col, diff_from="1979-2016"){
  #
  # Clean unwanted data
  #
  if (diff_from == "1979-2016"){
     unwanted_period <- "1950-2005"
     } else {
     unwanted_period <- "1979-2016"
  }
  dt <- dt %>% 
        filter(time_period != unwanted_period & 
               time_period != "2006-2025") %>% 
        data.table()

  if ("evap" %in% colnames(dt)){
    dt <- within(dt, remove(evap, runoff, base_flow, run_p_base))
  }
  if ("wtr_yr" %in% colnames(dt)) {
    dt <- within(dt, remove(wtr_yr))
  }
  dt <- within(dt, remove(year, day, precip))

  ################################################
  #
  # We need to do the following lines in order
  # to make the subtraction within groups work.
  #
  dt_hist <- dt %>% 
             filter(time_period == diff_from) %>%
             select(-c("emission")) %>%
             unique()%>%
             data.table()

  dt_hist$emission <- "hist"

  dt <- dt %>% filter(time_period != diff_from) %>% data.table()
  dt <- rbind(dt, dt_hist); rm(dt_hist)
  
  med_per_loc_mod_TP_em <- dt[, .( tgt_col = median(get(tgt_col))), 
                               by = c("location", "time_period", 
                                      "emission", "model", "month",
                                      "cluster")]

  setnames(med_per_loc_mod_TP_em, old="tgt_col", new="medians")

  median_diffs <- med_per_loc_mod_TP_em %>%
                  group_by(location, month) %>%
                  mutate(diff = medians - medians[time_period == diff_from])%>%
                  filter(time_period != diff_from)%>%
                  data.table()

  # to do percentages
  hist_meds_tl <- med_per_loc_mod_TP_em %>% 
                  filter(time_period == diff_from) %>% 
                  data.table()

  hist_meds_tl <- within(hist_meds_tl, remove(time_period, emission, model))
  setnames(hist_meds_tl, old=c("medians"), new="hist_median")

  median_diffs$hist_median <- median_diffs$medians -  median_diffs$diff
  median_diffs$perc_diff <- (median_diffs$diff * 100) / (median_diffs$hist_median)
  return(median_diffs)
}
####################################################################################
median_of_diff_of_medians <- function(dt){
  diffs <- dt %>%
           group_by(location, time_period, emission, cluster) %>%
           transmute(med_of_diffs_of_meds = median(diff))%>%
           unique() %>%
           data.table()

  perc_diffs <- dt %>%
                group_by(location, time_period, emission, cluster) %>%
                transmute(perc_med_of_diffs_of_meds = median(perc_diff))%>%
                unique() %>%
                data.table()

  diffs <- merge(diffs, perc_diffs, 
                 by=c("location", 'emission', 
                       "time_period", "cluster"))
  cols <- c("med_of_diffs_of_meds", "perc_med_of_diffs_of_meds")
  diffs[,(cols) := round(.SD, 1), .SDcols=cols]
  return(diffs)
}

median_diff_obs_or_modeled <- function(dt, tgt_col, diff_from){
  if (diff_from=="1979-2016"){
    median_diffs <- compute_median_diff(dt, tgt_col, diff_from="1979-2016")
    
    } else {
      # HERE we have: diff_from=="1950-2005"
      median_diffs <- data.table()
      
      # The followings is not necessary, it is done in compute... function
      dt <- dt %>% 
            filter(time_period != "2006-2025" & time_period != "1979-2016") %>% 
            data.table()
      all_mods <- unique(dt$model)
      for (mod in all_mods){
        curr_dt <- dt %>% filter(model == mod) %>% data.table()
        curr_dt <- compute_median_diff(curr_dt, tgt_col, diff_from="1950-2005")
        median_diffs <- rbind(median_diffs, curr_dt)
      }
  }
  return(median_diffs)
}

compute_median_diff <- function(dt, tgt_col, diff_from="1979-2016"){
  #
  # Clean unwanted data
  #
  if (diff_from == "1979-2016"){
    unwanted_period <- "1950-2005"
    } else {
    unwanted_period <- "1979-2016"
  }

  dt <- dt %>% 
        filter(time_period != unwanted_period & 
               time_period != "2006-2025") %>% 
        data.table()

  if ("evap" %in% colnames(dt)){
    dt <- within(dt, remove(evap, runoff, base_flow, run_p_base))
  }
  if ("wtr_yr" %in% colnames(dt)) {
    dt <- within(dt, remove(wtr_yr))
  }
  dt <- within(dt, remove(year, month, day, precip))

  ################################################
  #
  # We need to do the following lines in order
  # to make the subtraction within groups work.
  #
  dt_hist <- dt %>% 
             filter(time_period == diff_from) %>%
             select(-c("emission")) %>%
             unique()%>%
             data.table()

  dt_hist$emission <- "hist"

  dt <- dt %>% filter(time_period != diff_from) %>% 
        data.table()
  dt <- rbind(dt, dt_hist)
  
  # med_per_loc_mod_TP_em <- data.frame(dt) %>% 
  #                          group_by(location, time_period, 
  #                                    emission, model) %>% 
  #                          summarise(medians = median(get(tgt_col)))  %>% 
  #                          data.table()
  # print(colnames(dt))
  med_per_loc_mod_TP_em <- dt[, .( tgt_col = median(get(tgt_col))), 
                               by = c("location", "time_period", 
                                      "emission", "model", "cluster")]
  setnames(med_per_loc_mod_TP_em, old="tgt_col", new="medians")

  median_diffs <- med_per_loc_mod_TP_em %>%
                  group_by(location) %>%
                  mutate(diff = medians - medians[time_period == diff_from])%>%
                  data.table()

  median_diffs <- median_diffs %>% 
                  filter(time_period != diff_from)%>%
                  data.table()

  # to do percentages
  # hist_meds_tl <- med_per_loc_mod_TP_em %>% filter(time_period == diff_from) %>% data.table()
  # hist_meds_tl <- within(hist_meds_tl, remove(time_period, emission, model))
  # setnames(hist_meds_tl, old=c("medians"), new="hist_median")
  # median_diffs <- merge(median_diffs, hist_meds_tl, by="location", all.x=T)
  median_diffs$hist_median <- median_diffs$medians -  median_diffs$diff
  
  median_diffs$perc_diff <- (median_diffs$diff * 100) / (median_diffs$hist_median)
  return(median_diffs)
}

########################################################################
add_coord_from_location <- function(dt){
  x <- sapply(dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;
  return(dt)
}

month_numeric_2_str <- function(B){
  B$month <- as.character(B$month)
  B$month <- recode(B$month, "1" = "Jan", "2" = "Feb", "3" = "Mar",
                             "4" = "Apr", "5" = "May", "6" = "Jun", 
                             "7" = "Jul", "8" = "Aug", "9" = "Sep", 
                             "10" = "Oct", "11" = "Nov", "12" = "Dec")

  month_levels <- c("Sep", "Oct", "Nov", "Dec",
                    "Jan", "Feb", "Mar", "Apr", 
                    "May", "Jun", "Jul", "Aug")
  B$month <- factor(B$month, levels=month_levels)
  return(B)
}

# cluster_numeric_2_str <- function(B){
#   B$cluster <- as.character(B$cluster)
#   B$cluster <- recode(B$cluster, "5" = "5", 
#                                  "4" = "4",
#                                  "3" = "3",
#                                  "2" = "2",
#                                  "1" = "1")

#   categ_label <- c("1", "2", "3", "4", "5")  
#   B$cluster <- factor(B$cluster, levels=categ_label)

#   return(B)
# }

read_min_file <- function(conn){
  RLData <- readLines(conn)
  RLData_df <- as.data.frame(RLData)
  RLData_df <- stringr::str_split_fixed(RLData_df$RLData, "\t", 7)
  RLData_df <- data.table(RLData_df)

  col_name <- c("year", "month", "day", 
                "precip", "evap", "runoff", "base_flow")
  colnames(RLData_df) <- col_name
  return(RLData_df)
}
#########################################################
cluster_by_precip_elev <- function(observed_dt, scale=FALSE, no_clusters=4){
  #
  # max_clusters: maximum number of clusters to try and pick
  #               the best.
  #
  # for_elbow = data.table(no_clusters = c(1:max_clusters),
  #                        total_within_cluster_ss = rep(-666, max_clusters))
  set.seed(100)
  clust_data <- subset(observed_dt, select=c(elevation, ann_prec_mean))
  clusters_obj <- kmeans(clust_data, centers = no_clusters, nstart = 50)
  
  centroids <- data.table(clusters_obj$centers)
  centroids$cluster_label <- c(1:dim(centroids)[1])
  setnames(centroids, old=c("elevation", "ann_prec_mean"), 
                      new=c("elev_centriod", "prec_centroid"))
  
  # for_elbow[k, "total_within_cluster_ss"] <- clusters$betweenss
  clusters = data.table(location = observed_dt$location,
                        elevation = observed_dt$elevation,
                        ann_prec_mean = observed_dt$ann_prec_mean,
                        cluster_label = clusters_obj$cluster)
 # 1st method
  clusters <- merge(clusters, centroids)
  
  # re-order cluster labels so that the max label 
  # corresponds to max rain and so on
  # clusters <- clusters %>% 
  #             mutate("cluster" = frankv(centroid, 
  #                                       ties.method = "dense"))

  # clusters <- within(clusters, remove(cluster_label))
  setnames(clusters, old=c("cluster_label"), new=c("cluster"))
  return(list(clusters, clusters_obj))
}

cluster_by_elevation <- function(observed_dt, scale=FALSE, no_clusters=4){
  #
  # max_clusters: maximum number of clusters to try and pick
  #               the best.
  #
  # for_elbow = data.table(no_clusters = c(1:max_clusters),
  #                        total_within_cluster_ss = rep(-666, max_clusters))
  set.seed(100)
  clusters_obj <- kmeans(observed_dt$elevation, centers = no_clusters, nstart = 50)

  # for_elbow[k, "total_within_cluster_ss"] <- clusters$betweenss
  clusters = data.table(location = observed_dt$location,
                        elevation = observed_dt$elevation,
                        cluster_label = clusters_obj$cluster)
 # 1st method
  clusters <- clusters %>% 
              group_by(cluster_label) %>% #create the centroids variable
              mutate(centroid = mean(elevation)) %>%
              data.table()
  
  # re-order cluster labels so that the max label 
  # corresponds to max rain and so on
  clusters <- clusters %>% 
              mutate("cluster" = frankv(centroid, 
                                        ties.method = "dense"))

  clusters <- within(clusters, remove(cluster_label))
  return(list(clusters, clusters_obj))
}

cluster_yr_avging <- function(observed_dt, scale=FALSE, no_clusters=4){
  #
  # max_clusters: maximum number of clusters to try and pick
  #               the best.
  #
  if (scale == FALSE){
     observed_dt <- observed_dt %>% 
                   group_by(location) %>% 
                   summarise(target_col = mean(annual_cum_precip)) %>% 
                   data.table()
     } else {
      observed_dt <- observed_dt %>% 
                     group_by(location)%>% 
                     summarise(mean_annual_precip = mean(annual_cum_precip),
                               sd = sd(annual_cum_precip))%>%
                     mutate(target_col = (mean_annual_precip/sd)) %>% 
                     data.table()
  }
  # for_elbow = data.table(no_clusters = c(1:max_clusters),
  #                        total_within_cluster_ss = rep(-666, max_clusters))
  set.seed(100)
  clusters_obj <- kmeans(observed_dt$target_col, centers = no_clusters, nstart = 50)

  # for_elbow[k, "total_within_cluster_ss"] <- clusters$betweenss
  clusters = data.table(location = observed_dt$location,
                        ann_prec_mean = observed_dt$target_col,
                        cluster_label = clusters_obj$cluster)

  ##### Sort according descending order 4 is most rainy, 1 least
  # centroids_dt <- data.table(centroid=as.vector(clusters_obj$centers),
  #                            cluster_label = 1:no_clusters)
  # clusters <- merge(clusters, centroids_dt, 
  #                   by="cluster_label", all.x=TRUE)
  
  # 1st method
  clusters <- clusters %>% 
              group_by(cluster_label) %>% #create the centroids variable
              mutate(centroid = mean(ann_prec_mean)) %>%
              # ungroup() %>%
              data.table()
  
  # re-order cluster labels so that the max label 
  # corresponds to max rain and so on
  clusters <- clusters %>% 
              mutate("cluster" = frankv(centroid, ties.method = "dense"))

  clusters <- within(clusters, remove(cluster_label))
  return(list(clusters, clusters_obj))
}

#********************************************************
design_storm_4_allLoc_allMod_from_raw <- function(data_tbl, observed=FALSE){
  locations <- unique(data_tbl$location)
  models <- unique(data_tbl$model)
  
  no_locs <- length(locations)
  no_models <- length(models)
  emission <- unique(data_tbl$emission)

  ###################################################
  #
  # set up the output table:
  #
  if (observed==TRUE){
     mini_tab_nrow <- 1
     } else {
     mini_tab_nrow <- 4
  }
  n_rows <- no_locs * no_models * mini_tab_nrow
  col_names <- c("location", "model", "return_period", "five_years", "ten_years", 
                 "fifteen_years", "twenty_years", "twenty_five_years")
  final_table <- setNames(data.table(matrix(nrow = n_rows, 
                                     ncol = length(col_names))), col_names)

  final_table$location <- as.character(final_table$location)
  final_table$model <- as.character(final_table$model)
  final_table$return_period <- as.character(final_table$return_period)
  
  final_table$five_years <- as.numeric(final_table$five_years)
  final_table$ten_years <- as.numeric(final_table$ten_years)
  final_table$fifteen_years <- as.numeric(final_table$fifteen_years)
  final_table$twenty_years <- as.numeric(final_table$twenty_years)
  final_table$twenty_five_years <- as.numeric(final_table$twenty_five_years)
  ###################################################
  row_pointer <- 1
  
  for (loc in locations){
     for (mod in models){
       curr_dt <- data_tbl %>% filter(location==loc & model==mod)
       curr_storm <- design_storm_4_oneLoc_oneMod_from_raw(curr_dt, observed)
       final_table[row_pointer:(row_pointer + (mini_tab_nrow-1)), ] <- curr_storm
       row_pointer <- row_pointer + mini_tab_nrow
     }
  }
  return(final_table)
}

design_storm_4_oneLoc_oneMod_from_raw <- function(data_tbl, observed=FALSE){
  ################################################
  # This function is written to be applied to "an individual"
  # (location, model) pair
  # input : data_tbl has to have columns: 
  ################################################

  data_tbl <- find_annual_max_24_hr(data_tbl)
  data_tbl <- convert_precip_2_intens(data_tb=data_tbl,
                                      col_name="max_24_hr_precip_annual")
  
  # data_tbl <- put_time_period(data_tb=data_tbl, observed=obs)
  data_tbl <- design_storm_all_timePeriods(data_tbl)
  return(data_tbl)
}

design_storm_all_timePeriods <- function(data_tbl){
  ################################################################
  #
  # This function is written to be applied to "an individual"
  # (location, model)
  # input : data_tbl has to have columns: 
  #                  year, location, max_24_hr_precip_annual, 
  #                  max_24_hr_intens,
  #                  time_period.
  #
  ################################################################

  # initiate the table to be populated:
  data = data.table()

  time_periods <- unique(data_tbl$time_period)
  time_period_stats <- intensity_stats_by_time_period(data_tbl)

  for (time in time_periods){
    data_t = data_tbl %>% filter(time_period == time)
    stat_time <- time_period_stats %>% filter(time_period == time)
    new_row_vec = design_storm_4_1_time_period(data_tb = data_t, 
                                               avg = stat_time$mean, 
                                               std = stat_time$sd)
    new_row = data.table(return_period = time,
                         five_years = new_row_vec[1],
                         ten_years = new_row_vec[2],
                         fifteen_years = new_row_vec[3],
                         twenty_years = new_row_vec[4],
                         twenty_five_years = new_row_vec[5]
                         )
    data <- rbind(data, new_row)
  }
  col_n <- colnames(data)
  data$location <- unique(data_tbl$location)
  data$model <- unique(data_tbl$model)

  setcolorder(data, c("location", "model", col_n))
  return(data)
}

design_storm_4_1_time_period <- function(data_tb, avg, std){
  years_intervals <- c(5, 10, 15, 20, 25)
  gumbel_constants <- compute_gumbel_constant(years_intervals)
  storm_dens <- avg + (std * gumbel_constants)
  return(storm_dens)
}

compute_gumbel_constant <- function(n_years){
  return(-1 * (sqrt(6) / pi) * (0.5772 + log(log(n_years / (n_years-1) ))))
}

intensity_stats <- function(data_tb){
  stats <- data_tb[ , list(mean=mean(max_24_hr_intens), 
                           sd=sd(max_24_hr_intens)),
                     by = c("model", "emission")]
  return(stats)
}

intensity_stats_by_time_period <- function(data_tb){
  stats <- data_tb[ , list(mean = mean(max_24_hr_intens), 
                           sd = sd(max_24_hr_intens)), 
                      by = c("time_period", "model", "emission")]
  return(stats)
}

convert_precip_2_intens <- function(data_tb, col_name="max_24_hr_precip_annual"){
  # input:  data_tb: data of class data.table
  #         col_name: name of the column to be converted
  #                   to intensity. The colum containing
  #                   max_24_hr Event thing
  
  # output: data_tb with a new column added to it
  #         that is result of dividing column given by col_name
  #         by 24.
  data_tb$max_24_hr_intens <- data_tb[, get(col_name)]/24.00
  return(data_tb)
}

find_annual_max_24_hr <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output: a data table that has one row per                  #
  #         (location, year) with the max_24_hr_precip         #
  #         column in it.                                      #
  ##############################################################
  
  data_tb <- data_tb %>%
             group_by(year, location, model, emission, time_period) %>%
             summarise(max_24_hr_precip_annual = max(precip)) %>%
             data.table()
  return (data_tb)
}

find_monthly_max_24_hr <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #        precip_col_name: The name of the column containing  #
  #                         daily precipitation.               #
  #                                                            #
  # output: a data table that has one row per                  #
  #         (location, year) with the max_24_hr_precip         #
  #        column in it.                                       #
  ##############################################################
  
  data_tb <- data_tb %>%
             group_by(year, month, location, model, emission, time_period) %>%
             summarise(max_24_hr_precip_monthly = max(precip)) %>%
             data.table()
  return (data_tb)
}

find_chunk_max_24_hr <- function(data_tb, start_month, end_month){
  # input: data_tb the data containing precip column in it.
  #        start_month and end_month are the beginning and end of
  #        the period we are interested in like from 10 (oct) to 3 (March).
  #        
  # output: 
  #
  # Group by year, and create new calendar year, and do the thing.

  data_tb <- create_wtr_calendar(data_tb, wtr_yr_start=start_month)

  # get rid of unwanted months:
  if (start_chunk > end_chunk){
     data_tb <- data_tb %>% 
                filter(!(month %in% ((end_month+1):(start_month-1))))
     } else {
         data_tb <- data_tb %>%
                    filter(month %in% (start_month:end_month))
  }

  data_tb$chunk_month_max <- data_tb %>%
                             group_by(wtr_yr, location, model, emission, time_period) %>%
                             summarise(max_24_hr_precip_chunky = max(precip)) %>%
                             data.table()
  return(data_tb)
}

########################################################################
#
#        Cumulation of precipitation section
#
########################################################################
compute_chunky_cum <- function(data_tb, start_month, end_month){
  #
  # first, put water_calendar so proper months are grouped together
  # second, toss out unwanted months.
  # third, find, cumu. over a wtr_year
  #
  data_tb <- create_wtr_calendar(data_tb, wtr_yr_start = start_month)

  data_tb <- data_tb %>%
             filter(!(month %in% ((end_month+1):(start_month-1))))%>%
             data.table()

  if ("runoff" %in% colnames(data_tb)){
    data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(chunk_cum_runbase = cumsum(run_p_base)) %>%
             # slice(n()) %>%
             data.table()
  } else{
    data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(chunk_cum_precip = cumsum(precip)) %>%
             # slice(n()) %>%
             data.table()
  }
 return(data_tb)
}

# **********************************************************************
compute_wtr_yr_cum <- function(data_tb){
  #
  # input: data_tb has to have the water_year column in it
  # output: cumulative precip in each water_year
  #
  if ("runoff" %in% colnames(data_tb)){
    data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(annual_cum_runbase = cumsum(run_p_base)) %>%
             # slice(n()) %>%
             data.table()
  } else {
    data_tb <- data_tb %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             mutate(annual_cum_precip = cumsum(precip)) %>%
             # slice(n()) %>%
             data.table()
  }
  return (data_tb)
}
# **********************************************************************

compute_annual_cum <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output:                                                    #
  #                                                            #
  #                                                            #
  ##############################################################
  if ("run_p_base" %in% colnames(data_tb)){
  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_runbase = cumsum(run_p_base)) %>%
             # filter(month==12 & day==31) %>%
             data.table()
    } else {
      data_tb <- data_tb %>%
                 group_by(location, year, model, emission, time_period) %>%
                 mutate(annual_cum_precip = cumsum(precip)) %>%
                 # filter(month==12 & day==31) %>%
                 data.table()
  }
  return (data_tb)
}
# **********************************************************************

compute_monthly_cum <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output:                                                    #
  #                                                            #
  #                                                            #
  ##############################################################
  if ("runoff" %in% colnames(data_tb)){
    data_tb <- data_tb %>%
               group_by(location, year, month, model, emission) %>%
               mutate(monthly_cum_runbase = cumsum(run_p_base)) %>%
               # slice(which.max(day)) %>%
               data.table()
    } else {
      data_tb <- data_tb %>%
               group_by(location, year, month, model, emission) %>%
               mutate(monthly_cum_precip = cumsum(precip)) %>%
               # slice(which.max(day)) %>%
               data.table()
  }
  return (data_tb)
}
# **********************************************************************

########################################################################
#
#        Create Water Calendar
#
########################################################################
create_wtr_calendar <- function(data_tb, wtr_yr_start){
  l = unique(data_tb$time_period)
  all_dt <- data.table()
  for (tp in l){
    cr_dt <- data_tb %>% filter(time_period==tp) %>% data.table()
    cr_dt <- create_wtr_calendar_1_tp(cr_dt, wtr_yr_start)
    all_dt <- rbind(all_dt, cr_dt)
  }
  return(all_dt)
}

create_wtr_calendar_1_tp <- function(data_tb, wtr_yr_start){
  # input:
  # output:
  data_tb <- data_tb %>%
             mutate(wtr_yr = case_when(# If Jan:Sept then part of H2O yr of prev year - current year
                                       month %in% c(1:(wtr_yr_start-1)) ~ paste0("wtr_", (year - 1), "-", year),
                                       
                                       # If Oct:Dec then part of H2O yr of current year - next year
                                       month %in% c(wtr_yr_start:12) ~ paste0("wtr_", year, "-", (year + 1))
                                       )
                    ) %>%
             data.table()

  # cut the beginning and end of the data that do not belong
  # to any water year calendar!
  data_tb <- data_tb %>% 
             filter(!(year == min(data_tb$year) & (month < wtr_yr_start))) %>%
             filter(!(year == max(data_tb$year) & (month >= wtr_yr_start))) %>%
             data.table()
  return(data_tb)
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
                        ) %>%
                 data.table()
  }
  return(data_tb)
}

#####################################
#####################################    Not Used
#####################################

cluster_yr_time_series <- function(observed_dt, no_clusters=4){
  observed_dt <- subset(observed_dt, select=c(location, 
                                              year, 
                                              annual_cum_precip))
  
  observed_dt <- reshape(observed_dt, idvar = "location", 
                                      timevar = "year", 
                                      direction = "wide")
  locations <- observed_dt$location
  observed_dt <- within(observed_dt, remove(location))

  ts_clusters <- kmeans(observed_dt, centers = no_clusters, nstart = 25)
  return(ts_clusters)
}

make_percentage_column_discrete <- function(dta){
  tps <- unique(dta$time_period)
  emissions <- c("RCP 4.5", "RCP 8.5")
  result <- data.table()
  for (tp in tps){
    for (em in emissions){
      dt <- dta %>% filter(emission == em & time_period==tp) %>% data.table()
      minn <- min(dt$perc_diff)
      maxx <- max(dt$perc_diff)
      # c(0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.20, 0.40, 0.80)
      dt$perc_diff <- dt$perc_diff/100

      dt <- dt %>% 
            mutate(tagt_perc = case_when((perc_diff >= 0   & perc_diff <= 0.05) ~ 5,
                                         (perc_diff > 0.05 & perc_diff <= 0.1)  ~ 10,
                                         (perc_diff > 0.1  & perc_diff <= 0.15) ~ 15,
                                         (perc_diff > 0.15 & perc_diff <= 0.2)  ~ 20,
                                         (perc_diff > 0.2  & perc_diff <= 0.5)  ~ 50,
                                         (perc_diff > 0.5) ~ 100,
                                         (perc_diff < 0     & perc_diff >= -0.05) ~ -5,
                                         (perc_diff < -0.05 & perc_diff >= -0.1)  ~ -10,
                                         (perc_diff < -0.1  & perc_diff >= -0.15) ~ -15,
                                         (perc_diff < -0.15 & perc_diff >= -0.2)  ~ -20,
                                         (perc_diff < -0.2  & perc_diff >= -0.5)  ~ -50,
                                         (perc_diff < -.5) ~ -100
                                         )
                  ) %>%
            data.table()
    result <- rbind(result, dt)

    }
  }
  return(result)
}