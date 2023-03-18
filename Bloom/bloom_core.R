
convert_zero_to_365 <- function(dt){
  dt$thresh_75[dt$thresh_75==0] <- 365
  dt$thresh_70[dt$thresh_75==0] <- 365
  dt$thresh_65[dt$thresh_75==0] <- 365
  dt$thresh_60[dt$thresh_75==0] <- 365
  dt$thresh_55[dt$thresh_75==0] <- 365
  dt$thresh_50[dt$thresh_75==0] <- 365
  dt$thresh_45[dt$thresh_75==0] <- 365
  dt$thresh_40[dt$thresh_75==0] <- 365
  dt$thresh_35[dt$thresh_75==0] <- 365
  dt$thresh_30[dt$thresh_75==0] <- 365
  dt$thresh_25[dt$thresh_75==0] <- 365
  dt$thresh_20[dt$thresh_75==0] <- 365
  return(dt)
}

change_thresh_4_bloom <- function(dt, location_ls){
  #
  #  emissions are saved as rcp45, time periods are 2026-2075
  #  here we modify those stuff to make the file proper for 
  #  time series plots. 
  #
  dt <- within(dt, remove(start, 
                          sum, sum_J1, sum_F1, 
                          sum_M1, sum_A1, year))

  # make time_period similar to that of frost and bloom
  dt <- convert_thresh_TPs_to_three(dt)
  
  #
  # there are 8 locations in Montana here, drop them
  #
  dt <- dt %>% filter(location %in% location_ls) %>% data.table()
  
  thresh_F <- dt %>% filter(time_period=="future")
  thresh_F_85 <- thresh_F %>% filter(emission == "rcp85")
  thresh_F_45 <- thresh_F %>% filter(emission == "rcp45")
  thresh_F_85$emission <- "RCP 8.5"
  thresh_F_45$emission <- "RCP 4.5"
  thresh_F <- rbind(thresh_F_85, thresh_F_45)
  rm(thresh_F_85, thresh_F_45)

  thresh_mod_hist <- dt %>% filter(time_period=="modeled_hist")
  thresh_mod_hist_85 <- thresh_mod_hist
  thresh_mod_hist_45 <- thresh_mod_hist
  thresh_mod_hist_85$emission <- "RCP 8.5"
  thresh_mod_hist_45$emission <- "RCP 4.5"
  thresh_mod_hist <- rbind(thresh_mod_hist_85, thresh_mod_hist_45)
  rm(thresh_mod_hist_85, thresh_mod_hist_45)

  thresh_obs <- dt %>% filter(time_period=="observed")
  thresh_obs_85 <- thresh_obs
  thresh_obs_45 <- thresh_obs
  thresh_obs_85$emission <- "RCP 8.5"
  thresh_obs_45$emission <- "RCP 4.5"
  thresh_obs <- rbind(thresh_obs_85, thresh_obs_45)
  rm(thresh_obs_85, thresh_obs_45)

  thresh <- rbind(thresh_F, thresh_obs, thresh_mod_hist)
  rm(thresh_F, thresh_obs, thresh_mod_hist)
  return(thresh)
}

convert_thresh_TPs_to_three <- function(dt){
  FTP <- c("2006-2025", "2026-2050", "2051-2075", "2076-2099")

  dt_F <- dt %>% 
          filter(time_period %in% FTP)%>% 
          data.table()
  dt_F$time_period <- "future"

  dt_MH <- dt %>% 
          filter(time_period == "1950-2005")%>% 
          data.table()
  dt_MH$time_period <- "modeled_hist"

  dt_obs <- dt %>% 
          filter(time_period == "1979-2015")%>% 
          data.table()
  dt_obs$time_period <- "observed"
  dt <- rbind(dt_F, dt_MH, dt_obs)
  return(dt)
}

add_location <- function(dt){
  if (!("location" %in% colnames(dt))){
    dt$location <- paste0(dt$lat, "_", dt$long)
  }
  return(dt)
}

pick_obs_and_F <- function(dt){
  # just keep observed and future
  dt <- toss_model_hist_by_TP(dt)
  
  # drop 2006-2025
  F <- dt %>% 
       filter(chill_season >= "chill_2026-2027")%>% 
       data.table()

  obs <- pick_observed_by_TP(dt)
  dt <- rbind(F, obs)
  return(dt)
}

toss_F0 <- function(dt){
  F <- pick_future_by_TP(dt)
  F <- pick_F1_F3_by_chill_season(F)
  mod_hist <- pick_model_hist_by_TP(dt)
  obs <- pick_observed_by_TP(dt)
  dt <- rbind(F, mod_hist, obs)
  return(dt)
}

pick_F1_F3_by_chill_season <- function(dt){
  dt <- dt %>% 
        filter(chill_season >= "chill_2025-2026") %>% 
        data.table()
  return(dt)
}

pick_future_by_TP <- function(dt){
  dt <- dt %>% 
        filter(time_period == "future") %>% 
        data.table()
  return(dt)
}

pick_model_hist_by_TP <- function(dt){
  dt <- dt %>% 
        filter(time_period == "modeled_hist") %>% 
        data.table()
  return(dt)
}

pick_observed_by_TP <- function(dt){
  dt <- dt %>% 
        filter(time_period %in% c("observed", "Observed", "1979-2016")) %>% 
        data.table()
  return(dt)
}

toss_observed_by_TP <- function(dt){
  dt <- dt %>% 
        filter(time_period != "observed") %>% 
        data.table()
  return(dt)
}

toss_model_hist_by_TP <- function(dt){
  dt <- dt %>% 
        filter(time_period != "modeled_hist") %>%
        data.table()
  return(dt)
}

find_1st_frost <- function(data_dt){
  data_dt <- subset(data_dt, select = c("lat", "long", "tmin",
                                        "model", "chill_dayofyear", 
                                        "chill_season", "emission"))
  data_dt <- data_dt[tmin <= 0, ]
  data_dt <- data_dt[, .SD[which.min(chill_dayofyear)], 
                       by = list(lat, long, chill_season, 
                                 model, emission)]  
  return (data_dt)
}

convert_doy_to_chill_doy <- function(dt){
  dt$chill_doy <- 1
  for (m in (1:8)){
    dt$chill_doy[dt$month==m] <- dt$dayofyear[dt$month==m] + 122
  }
  
  for (m in (9:12)){
    dt$chill_doy[dt$month==m] <- dt$dayofyear[dt$month==m] - 244
  }
  return(dt)
}

add_chill_sept_DoY <- function(dt){
  # This function would work when all days and months
  # of a given year are there.
  # Otherwise, cumsum would not work, and you have to use
  # convert_doy_to_chill_doy(.) function.
  #

  # relabel the months so we can
  # sort them, and create day of year 
  # in the chill calendar
  dt <- mutate(dt, 
               chill_month=case_when(month == 1 ~ 13,
                                     month == 2 ~ 14,
                                     month == 3 ~ 15,
                                     month == 4 ~ 16,
                                     month == 5 ~ 17,
                                     month == 6 ~ 18,
                                     month == 7 ~ 19,
                                     month == 8 ~ 20,
                                     month == 9 ~ 9,
                                     month == 10 ~ 10,
                                     month == 11 ~ 11,
                                     month == 12 ~ 12))
  dt <- data.table(dt)
  dt$chill_dayofyear <- 1
  if ("emission" %in% colnames(dt)){
    dt[, chill_dayofyear := cumsum(chill_dayofyear), 
       by=list(chill_season, model, lat, long, emission)]
  } else {
    dt[, chill_dayofyear := cumsum(chill_dayofyear), 
         by=list(chill_season, model, lat, long)]
  }
  
  return(dt)
}

trim_chill_calendar <- function(dt){
  # gets rid of the first and last
  # incomplete chill years.
  minn <- min(unique(dt$chill_season))
  maxx <- max(unique(dt$chill_season))
  dt <- dt %>% 
        filter(chill_season != minn) %>% 
        data.table()
  dt <- dt %>% 
        filter(chill_season != maxx)%>% 
        data.table()
  return(dt)
}

put_chill_calendar <- function(data_tb, chill_start="sept"){
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

bloom_per_year_median_accross_models <- function(data_dt) {
  data_dt <- data_dt[, .(medDoY = as.integer(median(dayofyear))), 
                       by = c("lat", "long", "emission",
                              "fruit_type", "year")]
  return(data_dt)
}


bloom_medians_across_models_time_periods <- function(data){
  data <- data[, .(medDoY = as.integer(median(dayofyear))), 
               by=c("lat", "long", "fruit_type")]
  return (data)
}


generate_vertdd <- function(data_tb, lower_temp=6.1, upper_temp=25.9){
  #
  #
  # Oct 12. Meeting With Vince. 
  #
  # Second update of parameters.
  #
  # We changed the lower, upper temps and also the mean and sd of apples
  # based on the file "Daymet vs AWN summary.docx" from Vince's DropBox.
  # 
  # lower_temp = 4.5   <- old. first update since CodMoth
  # upper_temp = 24.28 <- old. first update since CodMoth
  #

  data_tb <- data.table(data_tb)
  lower = lower_temp; upper = upper_temp
  twopi = 2 * pi; pihlf = 0.5 * pi

  data_tb$summ = data_tb$tmin + data_tb$tmax 
  data_tb$diff = data_tb$tmax - data_tb$tmin
  data_tb$diffsq = data_tb$diff * data_tb$diff

  data_tb$b <- 2 * upper - data_tb$summ
  data_tb$bsq <- data_tb$b * data_tb$b
  data_tb$a <- 2 * lower - data_tb$summ
  data_tb$asq <- data_tb$a * data_tb$a
  data_tb$th1 <- atan(data_tb$a / sqrt(data_tb$diffsq - data_tb$asq))
  data_tb$th2 <- atan(data_tb$b / sqrt(data_tb$diffsq - data_tb$bsq))

  data_tb[tmin >= lower & tmax > upper, 
          vertdd:=((-diff*cos(th2)-a*(th2+pihlf))/twopi)]
  
  data_tb[tmin >= lower & tmax <= upper, 
          vertdd := summ/2 - lower]
  
  data_tb[tmin < lower & tmax <= upper, 
          vertdd:=(diff*cos(th1)-(a*(pihlf-th1)))/twopi]
  
  data_tb[tmin < lower & tmax > upper, 
          vertdd:=(-diff*(cos(th2)-cos(th1))-(a*(th2-th1)))/twopi]
  
  data_tb[tmin>tmax | tmax<=lower | tmin>=upper, vertdd:=0]
  
  data_tb <- within(data_tb, remove(summ, diff, diffsq, b,
                                    bsq, a, asq, th1, th2))
  
  data_tb = data_tb[, vert_Cum_dd := cumsum(vertdd),
                    by=list(lat, long, model, year)]
  data_tb$vert_Cum_dd_F = data_tb$vert_Cum_dd * 1.8

  # the following is updated from new word document of Vince (Daymet vs AWN summary.docx)
  # for the first updated ones after the codling moth.
  # cripps pink
  data_tb$cripps_pink = pnorm(data_tb$vert_Cum_dd_F, 
                              # the commented out params below are old from first update since CodMoth:
                              # mean = 436.61, sd = 52.58, # commented out on Oct 12, 2020
                              # mean = 367.83, sd = 51.69, # Changed on Oct 12, 2020. These are AgWeather Net
                              mean = 342.32, sd = 45.71, # Changed on Nov 17, 2020. These are Daymet data
                              lower.tail = TRUE)
  # Gala
  data_tb$gala = pnorm(data_tb$vert_Cum_dd_F,
                       # the commented out params below are old from first update since CodMoth:
                       # mean = 468.99, sd = 49.49, # commented out on Oct 12, 2020
                       # mean = 403.17, sd = 48.93, # Changed on Oct 12, 2020, These are AgWeather Net
                       mean = 374.18, sd = 42.49,   # Changed on Nov 17, 2020. These are Daymet data
                       lower.tail = TRUE)
  # Red Deli
  data_tb$red_deli = pnorm(data_tb$vert_Cum_dd_F, 
                           # the commented out params below are old from first update since CodMoth:
                           # mean = 465.90, sd = 53.87, # commented out on Oct 12, 2020 
                           # mean = 400.67, sd = 51.41, # Changed on Oct 12, 2020, These are AgWeather Net.
                           mean = 361.67, sd = 46.57, # Changed on Nov 17, 2020. These are Daymet data
                           lower.tail = TRUE)
  return(data_tb)
}


generate_vertdd_for_sensitivity <- function(data_tb, lower_temp=6.1, upper_temp=25.9, distribution_mean, start_doy){
  data_tb <- data.table(data_tb)
  lower = lower_temp; upper = upper_temp
  twopi = 2 * pi; pihlf = 0.5 * pi

  data_tb$summ = data_tb$tmin + data_tb$tmax 
  data_tb$diff = data_tb$tmax - data_tb$tmin
  data_tb$diffsq = data_tb$diff * data_tb$diff

  data_tb$b <- 2 * upper - data_tb$summ
  data_tb$bsq <- data_tb$b * data_tb$b
  data_tb$a <- 2 * lower - data_tb$summ
  data_tb$asq <- data_tb$a * data_tb$a
  data_tb$th1 <- atan(data_tb$a / sqrt(data_tb$diffsq - data_tb$asq))
  data_tb$th2 <- atan(data_tb$b / sqrt(data_tb$diffsq - data_tb$bsq))

  data_tb[tmin >= lower & tmax > upper, 
          vertdd:=((-diff*cos(th2)-a*(th2+pihlf))/twopi)]
  
  data_tb[tmin >= lower & tmax <= upper, 
          vertdd := summ/2 - lower]
  
  data_tb[tmin < lower & tmax <= upper, 
          vertdd:=(diff*cos(th1)-(a*(pihlf-th1)))/twopi]
  
  data_tb[tmin < lower & tmax > upper, 
          vertdd:=(-diff*(cos(th2)-cos(th1))-(a*(th2-th1)))/twopi]
  
  data_tb[tmin>tmax | tmax<=lower | tmin>=upper, vertdd:=0]
  
  data_tb <- within(data_tb, remove(summ, diff, diffsq, b,
                                    bsq, a, asq, th1, th2))
  #
  #  Toss the first few days in Jan
  #
  print ("line 456")
  print (colnames(data_tb))
  
  if (start_doy>1){
    data_tb <- data_tb %>% 
               group_by(lat, long, year, emission, model)%>% 
               slice(-1:-(start_doy-1)) %>% 
               data.table()
  }
  
  data_tb = data_tb[, vert_Cum_dd := cumsum(vertdd),
                    by=list(lat, long, model, year)]

  data_tb$vert_Cum_dd_F = data_tb$vert_Cum_dd * 1.8

  # the following is updated from new word document of Vince
  # for the old ones see the codling moth core.
  # cripps pink
  data_tb$cripps_pink = pnorm(data_tb$vert_Cum_dd_F, 
                              mean = distribution_mean, sd = 52.58, 
                              lower.tail = TRUE)
  return(data_tb)
}



cumGDD_cut_at_50PercBloom <- function(data, cut_off){

  data$dayofyear <- 1 # dummy
  data[, dayofyear := cumsum(dayofyear), by=list(year, lat, long, model, emission)]


  data_bloom <- subset(data, select = c("year", "month", "day", "dayofyear",
                                        "lat", "long", "model", "emission",
                                        "cripps_pink", "gala", "red_deli"))

  data_cumVertGDD <- subset(data, select = c("year", "month", "day", "dayofyear",
                                             "lat", "long", "model", "emission",
                                             "vert_Cum_dd_F"))
  
  data_bloom <- melt(data_bloom, 
                     id.vars = c("lat", "long", "model", "emission",
                                 "year", "month", "day", "dayofyear"),
                     variable.name = "fruit_type")

  setnames(data_bloom, old=c("value"), new=c("bloom_perc"))

  # left join to get the vertical GDD back into the data table.

  data_bloom <- data.table(dplyr::left_join(x = data_bloom, y = data_cumVertGDD))
  data_bloom = data_bloom[bloom_perc >= cut_off, ]
  data_bloom = data_bloom[, head(.SD, 1), 
                          by = c("lat", "long", "model", "emission",
                          "year", "fruit_type")]
  
  return (data_bloom)
}

bloom_cut_off <- function(data, cut_off){
  data <- subset(data, select = c("year", "month", "day",
                                  "lat", "long", 
                                  "model", "emission",
                                  "cripps_pink", "gala", "red_deli"))
  
  data$dayofyear <- 1 # dummy
  data[, dayofyear := cumsum(dayofyear), 
       by=list(year, lat, long, model, emission)]
  
  data <- melt(data, id.vars = c("lat", "long", 
                                 "model", "emission",
                                 "year", "month", "day", 
                                 "dayofyear"),
               variable.name = "fruit_type")
  setnames(data, old=c("value"), new=c("bloom_perc"))
  data = data[bloom_perc >= cut_off, ]
  data = data[, head(.SD, 1), 
                by = c("lat", "long", "model", "emission",
                       "year", "fruit_type")]
  
  return (data)
}