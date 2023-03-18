rm(list=ls())
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_1)
source(source_path_2)
#############################################
data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/"
data_dir <- paste0(data_dir, "chilling/frost_bloom/Feb/")
#############################################

six_models <- c("BNU-ESM", "CanESM2", "GFDL-ESM2G", 
                "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")

emissions <- c("RCP 4.5", "RCP 8.5")
cities <- c("Hood River", "Walla Walla", "Richland", 
	          "Yakima", "Wenatchee", "Omak")
apple_types <- c("Cripps Pink", "Gala", "Red Deli") # 
fruit_types <- c("apple", "Cherry", "Pear")
thresholds <- c("threshold: 70", "threshold: 75")
time_window <- c("2070_2094")

vect <- c("city", "model", "emission", "thresh", 
          "time_window", "fruit_type")

fruit <- fruit_types[1]

for (fruit in fruit_types){
  full_cripps <- CJ(cities, six_models, emissions, thresholds, time_window, c(paste0(fruit, ": Cripps Pink")))
  full_gala <- CJ(cities, six_models, emissions, thresholds, time_window, c(paste0(fruit, ": Gala")))
  full_red <- CJ(cities, six_models, emissions, thresholds, time_window, c(paste0(fruit, ": Red Deli")))
  
  setnames(full_cripps, old=paste0("V", 1:6), new=vect)
  setnames(full_gala, old=paste0("V", 1:6), new=vect)
  setnames(full_red, old=paste0("V", 1:6), new=vect)

  all_data <- read.csv(paste0(data_dir, fruit, "_crossovers.csv"), 
                       as.is=TRUE)
  
  cripps <- all_data %>% 
            filter(fruit_type == paste0(fruit, ": Cripps Pink")) %>% 
            data.table()

  gala <- all_data %>% 
          filter(fruit_type == paste0(fruit, ": Gala")) %>%
          data.table()

  red <- all_data %>% 
         filter(fruit_type == paste0(fruit, ": Red Deli")) %>% 
         data.table()

  cripps <- merge(full_cripps, cripps, all.x=TRUE)
  gala <- merge(full_gala, gala, all.x=TRUE)
  red <- merge(full_red, red, all.x=TRUE)

  cripps[is.na(cripps)] <- 0
  gala[is.na(gala)] <- 0
  red[is.na(red)] <- 0

  cripps <- cripps[order(city, model, emission, thresh, fruit_type), ]
  gala <- gala[order(city, model, emission, thresh, fruit_type), ]
  red <- red[order(city, model, emission, thresh, fruit_type), ]

  cripps_70_85 <- cripps %>% filter(thresh=="threshold: 70" & emission=="RCP 8.5")
  cripps_70_45 <- cripps %>% filter(thresh=="threshold: 70" & emission=="RCP 4.5")
  cripps_70 <- rbind(cripps_70_85, cripps_70_45)

  cripps_75_85 <- cripps %>% filter(thresh=="threshold: 75" & emission=="RCP 8.5")
  cripps_75_45 <- cripps %>% filter(thresh=="threshold: 75" & emission=="RCP 4.5")
  cripps_75 <- rbind(cripps_75_85, cripps_75_45)

  cripps <- rbind(cripps_75, cripps_70)

  red_70_85 <- red %>% filter(thresh=="threshold: 70" & emission=="RCP 8.5")
  red_70_45 <- red %>% filter(thresh=="threshold: 70" & emission=="RCP 4.5")
  red_70 <- rbind(red_70_85, red_70_45)

  red_75_85 <- red %>% filter(thresh=="threshold: 75" & emission=="RCP 8.5")
  red_75_45 <- red %>% filter(thresh=="threshold: 75" & emission=="RCP 4.5")
  red_75 <- rbind(red_75_85, red_75_45)

  red <- rbind(red_75, red_70)

  gala_70_85 <- gala %>% filter(thresh=="threshold: 70" & emission=="RCP 8.5")
  gala_70_45 <- gala %>% filter(thresh=="threshold: 70" & emission=="RCP 4.5")
  gala_70 <- rbind(gala_70_85, gala_70_45)

  gala_75_85 <- gala %>% filter(thresh=="threshold: 75" & emission=="RCP 8.5")
  gala_75_45 <- gala %>% filter(thresh=="threshold: 75" & emission=="RCP 4.5")
  gala_75 <- rbind(gala_75_85, gala_75_45)

  gala <- rbind(gala_75, gala_70)

  write.table(cripps, 
              file = paste0(data_dir, fruit, "_cripps_crossovers.csv"), 
              row.names=FALSE, na="", col.names=TRUE, sep=",")
  
  write.table(gala, 
              file = paste0(data_dir, fruit, "_gala_crossovers.csv"), 
              row.names=FALSE, na="", col.names=TRUE, sep=",")

  write.table(red, 
              file = paste0(data_dir, fruit, "_red_crossovers.csv"), 
              row.names=FALSE, na="", col.names=TRUE, sep=",")
}



