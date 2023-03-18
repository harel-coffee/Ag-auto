rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)


hardiness_core <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/2021_for_paper/hardiness_core.R"
source(hardiness_core)

########################################
########
########   directories
########
########################################
JulienOutput_dir_base <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/is_output_of_step_3/"
param_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/"
output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/daily_data/"

########################################

locations <- data.table(read.csv(paste0(param_dir, "limited_locations_CH.csv")))
locations$location <- paste0(locations$lat, "_", locations$long)

grape_varieties <- c("cabernet_sauvignon", "chardonnay", "merlot", "reisling")
models <- dir(paste0(JulienOutput_dir_base, grape_varieties[1]))
emissions <- c("rcp45", "rcp85")

needed_columns <- c("variety", "year", "Date", "jday", "t_max", "t_min", 
                    "predicted_Hc", "CDI", "hardiness_year")
#####
#####
#####
print (grape_varieties)
# grape_varieties <- grape_varieties[1]


for (a_variety in grape_varieties){
  CH_data <- data.table()
  for (a_model in models){
    if (a_model == "historical"){
      for (a_location in locations$location){
         file_pathName <- paste0(JulienOutput_dir_base, a_variety, "/", a_model, "/", "data_", a_location, ".csv.gz")
         curr_ch <- data.table::fread(file_pathName)

         curr_ch <- curr_ch[!is.na(curr_ch$hardiness_year),]
         curr_ch <- subset(curr_ch, select=needed_columns)

         curr_ch$location <- a_location
         curr_ch$model <- "Observed"
         curr_ch$emission <- "Observed"
         curr_ch$time_period <- "Observed"
         curr_ch <- add_CH_DoY(curr_ch)

         CH_data <- rbind(CH_data, curr_ch)
        }

      } else{
      for (emission in emissions){
        for (a_location in locations$location){
           file_pathName <- paste0(JulienOutput_dir_base, a_variety, "/", a_model, "/", emission, "/data_", a_location, ".csv.gz")
           curr_ch <- data.table::fread(file_pathName)

           curr_ch <- curr_ch[!is.na(curr_ch$hardiness_year),]
           curr_ch <- subset(curr_ch, select=needed_columns)

           curr_ch <- curr_ch %>%
                      filter(year >= 2024) %>%
                      data.table()

           curr_ch <- curr_ch %>%
                      mutate(time_period = case_when(hardiness_year %in% c(2026:2050) ~ "2026-2050",
                                                     hardiness_year %in% c(2051:2075) ~ "2051-2075",
                                                     hardiness_year %in% c(2076:2099) ~ "2076-2099")
                            )

           curr_ch$location <- a_location
           curr_ch$model <- a_model
           curr_ch$emission <- emission

           curr_ch <- add_CH_DoY(curr_ch)
           CH_data <- rbind(CH_data, curr_ch)
        }
      }
    }
  }

  sort_columns <- c("location", "variety", "Date", "year", "jday", "t_max", "t_min",
                    "predicted_Hc", "CDI", "hardiness_year", "model", "emission", "CH_DoY")

  CH_data <- subset(CH_data, select=sort_columns)

  fwrite(CH_data,  file=paste0(output_dir, "/all_", a_variety, ".csv.gz"), row.names=F, compress="gzip")
}




