library(data.table)
library(dplyr)
options(dplyr.summarise.inform = FALSE)

binary_source_path = "/home/h.noorazar/Sid/sidFabio/read_binary_core.R" # Kamiak
source(binary_source_path)
options(digits=9)

seasonal_weather_aggregate_oneLoc <- function(path_, file_name, start_year, end_year){

  #
  # Here we aggregate seasonal weather for one location and then we will
  # use them to aggregate over a county.
  #

  #
  # Season 1 average temperature (◦C) 
  # Season 2 (growing season) average temperature (◦C)
  # Season 3 average temperature (◦ C) 
  # Season 4 average temperature (◦ C) 

  # Season 1 total precipitation (mm) 
  # Season 2 (growing season) season total precipitation (mm)
  # Season 3 total precipitation (mm) 
  # Season 4 total precipitation (mm)
  #


  #
  # The following text is copied from compute_GDD_linear() from sidFabio_core.R
  #  on Kamiak everything has 8 variales
  #
  # future data are all over the place. West are in Adams directory
  # non-west are elsewhere.  Hence this if-else statement.
  # right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))
  #
  no_vars_ = 8
  
  # data_type_ is model type; observed modeled_hist, or GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
  # to use right time span of the binary files
  met_data <- read_binary_new(file_path = paste0(path_, file_name), 
                              no_vars = no_vars_,
                              start_year = start_year,
                              end_year = end_year)

  met_data$Tavg = (met_data$tmax + met_data$tmin)/2
  met_data <- add_seasons(met_data)
  
  # avg_Tavg = met_data[ , mean(Tavg), by = season]
  # setnames(avg_Tavg, old=c("V1"), new=c("avg_Tavg"))
  avg_Tavg_per_season <- met_data %>%
                         group_by(year, season) %>%
                         summarise(avg_Tavg = mean(Tavg)) %>% 
                         data.table()

  total_precip_per_season <- met_data %>%
                             group_by(year, season) %>%
                             summarise(total_precip = sum(precip)) %>% 
                             data.table()

  seasonal_df <- dplyr::left_join(x = avg_Tavg_per_season, y = total_precip_per_season, by = c("year", "season"))

  cols=c("avg_Tavg", "total_precip")
  seasonal_df[,(cols) := round(.SD, 3), .SDcols=cols]

  return (seasonal_df)
}



add_seasons <- function(data_tb){
  #   Seasons: Season 1: Jan - Mar
  #            Season 2: Apr - Jul
  #            Season 3: Aug - Sep
  #            Season 4: Oct - Dec
  data_tb <- data_tb %>%
             mutate(season = case_when(
                    month %in% c(1:3)   ~ "S1",
                    month %in% c(4:7)   ~ "S2",
                    month %in% c(8:9)   ~ "S3",
                    month %in% c(10:12) ~ "S4")) %>%
             data.table()
  return (data_tb)
}


