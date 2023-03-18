rm(list=ls())
library(data.table)

### Adult Emergence
master_path = "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/scale_sensitivity/"
file_name = "combined_CM_"
version = "rcp85"

scale = "1"

file_name = paste0(master_path, scale, "/", file_name, version, ".rds" )
data <- data.table(readRDS(file_name))
data <- subset(data, select = c("Emergence", "ClimateGroup", 
                                "ClimateScenario", 
                                "CountyGroup"))

data$CountyGroup = as.character(data$CountyGroup)
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

data = data[, .(Emergence = Emergence),
              by = c("ClimateGroup", "CountyGroup")]

data <- subset(data, select = c("ClimateGroup", "CountyGroup", "Emergence"))
df <- data.frame(data)
df <- (df %>% group_by(CountyGroup, ClimateGroup))
medians <- (df %>% summarise(med = median(Emergence)))
medians
