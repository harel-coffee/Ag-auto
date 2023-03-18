#
#  Here we compute the vertDD based on chill_season calendar and pased on
#  whenever the CP trigger is on
#
#
rm(list=ls())
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)

#__________________________________________________________________________________
data_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

#__________________________________________________________________________________
trigger_dt <- readRDS(paste0(data_dir, "lowVariety/heatTriggers_sept_summary_comp_lowVariety.rds"))
heatAccum <- readRDS(paste0(data_dir,  "bloom_01Step_4locations_for_chill.rds"))

heatAccum$location <- paste0(heatAccum$lat, "_", heatAccum$long)

heatAccum <- within(heatAccum, 
             remove(cripps_pink, gala, red_deli, vert_Cum_dd, vert_Cum_dd_F, lat, long))

#__________________________________________________________________________________
#
#  Some of the chill seasons at are not trimmed. Toss them. Harmonize the data
#

heatAccum <- heatAccum %>% 
             filter(time_period %in% c("1979-2016", "2026-2050", "2051-2075", "2076-2099")) %>% 
             data.table()

heatAccum <- heatAccum %>% 
             filter(chill_season %in% trigger_dt$chill_season) %>% 
             data.table()

#__________________________________________________________________________________

heatAccum_F_45 <- heatAccum %>% filter(emission %in% c("RCP 4.5")) %>% data.table()
heatAccum_F_85 <- heatAccum %>% filter(emission %in% c("RCP 8.5")) %>% data.table()
heatAccum_obse <- heatAccum %>% filter(emission %in% c("Observed")) %>% data.table()

locations <- unique(heatAccum_obse$location)
#__________________________________________________________________________________
#
#   set the vertDD = 0 before the trigger is kicked in
#

a_model = unique(heatAccum_F_45$model)[1]
a_chill_yr = unique(heatAccum_F_45$chill_season)[1]
aloc = locations[1]

#
#  Future RCP 4.5 loop
#
for (a_model in unique(heatAccum_F_45$model)){
  for (a_chill_yr in unique(heatAccum_F_45$chill_season)){
    for (aloc in locations){

      curr_trigger <- trigger_dt[model == a_model & chill_season == a_chill_yr & 
                                 location == aloc & emission == "RCP 4.5"]

      curr_trigger <- curr_trigger$thresh_37half

      heatAccum_F_45[model == a_model & chill_season == a_chill_yr & 
                     location == aloc & chill_dayofyear <= curr_trigger, vertdd:=0]

      
    }
  }
}

#
#  Future RCP 8.5 loop
#
for (a_model in unique(heatAccum_F_85$model)){
  for (a_chill_yr in unique(heatAccum_F_85$chill_season)){
    for (aloc in locations){

      curr_trigger <- trigger_dt[model == a_model & chill_season == a_chill_yr & 
                                 location == aloc & emission == "RCP 8.5"]

      curr_trigger <- curr_trigger$thresh_37half

      heatAccum_F_85[model == a_model & chill_season == a_chill_yr & 
                     location == aloc & chill_dayofyear <= curr_trigger, vertdd:=0]

    }
  }
}

#
# observed loop
#
for (a_model in unique(heatAccum_obse$model)){
  for (a_chill_yr in unique(heatAccum_obse$chill_season)){
    for (aloc in locations){

      curr_trigger <- trigger_dt[model == a_model & chill_season == a_chill_yr & 
                                 location == aloc & emission == "Observed"]

      curr_trigger <- curr_trigger$thresh_37half

      heatAccum_obse[model == a_model & chill_season == a_chill_yr & 
                     location == aloc & chill_dayofyear <= curr_trigger, vertdd:=0]

    }
  }
}

#__________________________________________________________________________________

heatAccum_new <- rbind(heatAccum_obse, heatAccum_F_45, heatAccum_F_85)

#__________________________________________________________________________________
#
#  Test
# 

a_model = unique(heatAccum_F_45$model)[6]
a_chill_yr = unique(heatAccum_F_45$chill_season)[6]
aloc = locations[1]

curr_trigger <- trigger_dt[model == a_model & chill_season == a_chill_yr & 
                           location == aloc & emission == "RCP 8.5"]

A <- heatAccum[model == a_model & chill_season == a_chill_yr & 
               location == aloc & emission == "RCP 8.5"]

A_new <- heatAccum_new[model == a_model & chill_season == a_chill_yr & 
                       location == aloc & emission == "RCP 8.5"]

View(A)
View(A_new)
#__________________________________________________________________________________
#
# compute the cum_vertDD
#
heatAccum_new = heatAccum_new[, vert_Cum_dd := cumsum(vertdd), by=list(location, chill_season, model, emission)]
heatAccum_new$vert_Cum_dd_F = heatAccum_new$vert_Cum_dd * 1.8

A <- heatAccum_new[model == a_model & chill_season == a_chill_yr & 
                   location == aloc & emission == "RCP 8.5"]
View(A)

#__________________________________________________________________________________

saveRDS(heatAccum_new, paste0(data_dir, "lowVariety/triggerBased_vertDD_lowVariety.rds"))

#__________________________________________________________________________________
#__________________________________________________________________________________
#__________________________________________________________________________________
#__________________________________________________________________________________
#
#
#       Bloom fit
#
heatAccum_F_45 <- heatAccum_new %>% filter(emission %in% c("RCP 4.5")) %>% data.table()
heatAccum_F_85 <- heatAccum_new %>% filter(emission %in% c("RCP 8.5")) %>% data.table()
heatAccum_obse <- heatAccum_new %>% filter(emission %in% c("Observed")) %>% data.table()

locations <- unique(heatAccum_obse$location)
#__________________________________________________________________________________
#
#   set the vertDD = 0 before the trigger is kicked in
#

a_model    = unique(heatAccum_F_45$model)[1]
a_chill_yr = unique(heatAccum_F_45$chill_season)[1]
aloc = locations[1]

#
#  Future RCP 4.5 loop
#
for (a_model in unique(heatAccum_F_45$model)){
  for (a_chill_yr in unique(heatAccum_F_45$chill_season)){
    for (aloc in locations){
      
      # the commented out params below are old from first update since CodMoth:
      # mean = 436.61, sd = 52.58, # commented out on Oct 12, 2020
      # mean = 367.83, sd = 51.69, # Changed on Oct 12, 2020. These are AgWeather Net
      # last Changed on Nov 17, 2020. These are Daymet data

      curr_dt <- heatAccum_F_45[model == a_model & chill_season == a_chill_yr & location == aloc]

      dist <- pnorm(curr_dt$vert_Cum_dd_F, mean = 342.32, sd =45.71, lower.tail = TRUE)
    
      heatAccum_F_45[model == a_model & chill_season == a_chill_yr & 
                     location == aloc, cripps_pink := dist]

      
    }
  }
}

#
#  Future RCP 8.5 loop
#
for (a_model in unique(heatAccum_F_85$model)){
  for (a_chill_yr in unique(heatAccum_F_85$chill_season)){
    for (aloc in locations){
      
      # the commented out params below are old from first update since CodMoth:
      # mean = 436.61, sd = 52.58, # commented out on Oct 12, 2020
      # mean = 367.83, sd = 51.69, # Changed on Oct 12, 2020. These are AgWeather Net
      # last Changed on Nov 17, 2020. These are Daymet data

      curr_dt <- heatAccum_F_85[model == a_model & chill_season == a_chill_yr & location == aloc]

      dist <- pnorm(curr_dt$vert_Cum_dd_F, mean = 342.32, sd =45.71, lower.tail = TRUE)
    
      heatAccum_F_85[model == a_model & chill_season == a_chill_yr & 
                     location == aloc, cripps_pink := dist]

      
    }
  }
}

#
# observed loop
#
for (a_model in unique(heatAccum_obse$model)){
  for (a_chill_yr in unique(heatAccum_obse$chill_season)){
    for (aloc in locations){
      
      # the commented out params below are old from first update since CodMoth:
      # mean = 436.61, sd = 52.58, # commented out on Oct 12, 2020
      # mean = 367.83, sd = 51.69, # Changed on Oct 12, 2020. These are AgWeather Net
      # last Changed on Nov 17, 2020. These are Daymet data

      curr_dt <- heatAccum_obse[model == a_model & chill_season == a_chill_yr & location == aloc]

      dist <- pnorm(curr_dt$vert_Cum_dd_F, mean = 342.32, sd =45.71, lower.tail = TRUE)
    
      heatAccum_obse[model == a_model & chill_season == a_chill_yr & 
                     location == aloc, cripps_pink := dist]

      
    }
  }
}

triggerBased_vertDD_CrippsBloom <- rbind(heatAccum_obse, heatAccum_F_45, heatAccum_F_85)
saveRDS(triggerBased_vertDD_CrippsBloom, paste0(data_dir, "lowVariety/triggerBased_vertDD_CrippsBloom_no50cut_lowVariety.rds"))

