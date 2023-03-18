rm(list=ls())
library(data.table)
# library(ggplot2)
library(ggpubr)
library(dplyr)
library(tidyverse)
library(lubridate)

start_time <- Sys.time()

input_dir = "/Users/hn/Documents/01_research_data/my_aeolus_2015/all_local/4_cumdd/"
# input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/"
version = c("rcp45", "rcp85")

setwd(input_dir)

color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")
vers <- version[2]

for (vers in version){
  print (vers)
  if (vers == "rcp45"){ plot_title <- "RCP 4.5"} else{ plot_title <- "RCP 8.5"}
  
  data = readRDS(paste0("./", "cumdd_CMPOP_", vers, ".rds"))
  data = subset(data, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumDDinF"))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  # add the new season column
  data[, season := as.character(0)]
  data[data[ , data$dayofyear <= 90]]$season = "Qr. 1"
  data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "Qr. 2"
  data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "Qr. 3"
  data[data[ , data$dayofyear >= 274]]$season = "Qr. 4"
  data = within(data, remove(dayofyear))
  data$season = factor(data$season, levels = c("Qr. 1", "Qr. 2", "Qr. 3", "Qr. 4"))

  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)
  
  
  df <- data.frame(data)
  df <- (df %>% group_by(ClimateGroup, CountyGroup, season))
  #
  # Medians
  #
  medians <- (df %>% summarise(CumDDinFMedians = median(CumDDinF)))
  medians <- data.table(medians)
  cols <- c("CumDDinFMedians")
  medians[, (cols) := round(.SD, 2), .SDcols=cols]
  if (vers == "rcp85"){em = "RCP 8.5"} else {em = "RCP 4.5"}
  medians$emission <- em
  assign(x = paste0("cumdd_table_medians_", vers), value ={medians})
  #
  # Range
  #
  mins <- (df %>% summarise(CumDDinFMin = min(CumDDinF)))
  mins <- data.table(mins)
  cols <- c("CumDDinFMin")
  mins[, (cols) := round(.SD, 2), .SDcols=cols]
  if (vers == "rcp85"){em = "RCP 8.5"} else {em = "RCP 4.5"}
  mins$emission <- em
  
  maxs <- (df %>% summarise(CumDDinFMax = max(CumDDinF)))
  maxs <- data.table(maxs)
  cols <- c("CumDDinFMax")
  maxs[, (cols) := round(.SD, 2), .SDcols=cols]
  if (vers == "rcp85"){em = "RCP 8.5"} else {em = "RCP 4.5"}
  maxs$emission <- em

  range <- merge(mins, maxs)
  range$range <- range$CumDDinFMax - range$CumDDinFMin

  assign(x = paste0("cumdd_table_range_", vers), value ={mins})
}

dd_median_table <- rbind(cumdd_table_medians_rcp85, cumdd_table_medians_rcp45)
dd_range_table <- rbind(cumdd_table_range_rcp85, cumdd_table_range_rcp45)


write.csv(dd_median_table, file = "./dd_median_table.csv", row.names = FALSE)
write.csv(dd_range_table, file = "./dd_range_table.csv", row.names = FALSE)


print( Sys.time() - start_time)







