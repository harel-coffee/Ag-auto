library(data.table)
library(ggplot2)

data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
#categories = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
categories = c("BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
file_prefix = "data_"
file_list = "list"
ClimateGroup = list("Historical", "2040's", "2060's", "2080's")
cellByCounty = data.table(read.csv(paste0(data_dir, "CropParamCRB.csv")))
#data = data.table()
conn = file(paste0(data_dir, file_list), open = "r")
locations = readLines(conn)
for( category in categories) {
  for( location in locations) {
    #print(location) 
    #print(category)
    #filename = paste0(category, "/", file_prefix, location)
    filename = paste0(category, "/rcp45/", file_prefix, location)
    #print(filename)
    
    if(category == "historical") {
      start_year = 1979
      end_year = 2015
    }
    else {
      start_year = 2006
      end_year = 2099
    }
    
    temp <- prepareData(filename, data_dir, start_year, end_year)
    temp_data <- data.table()
    if(category == "historical") {
      temp$ClimateGroup[temp$year >= 1979 & temp$year <= 2006] <- "Historical"
      temp_data <- rbind(temp_data, temp[temp$year >= 1979 & temp$year <= 2006, ])
    }
    else {
      temp$ClimateGroup[temp$year > 2025 & temp$year <= 2055] <- "2040's"
      temp_data <- rbind(temp_data, temp[temp$year > 2025 & temp$year <= 2055, ])
      temp$ClimateGroup[temp$year > 2045 & temp$year <= 2075] <- "2060's"
      temp_data <- rbind(temp_data, temp[temp$year > 2045 & temp$year <= 2075, ])
      temp$ClimateGroup[temp$year > 2065 & temp$year <= 2095] <- "2080's"
      temp_data <- rbind(temp_data, temp[temp$year > 2065 & temp$year <= 2095, ])
    }
    loc = tstrsplit(location, "_")
    temp_data$latitude <- as.numeric(unlist(loc[1]))
    temp_data$longitude <- as.numeric(unlist(loc[2]))
    temp_data$County <- as.character(unique(cellByCounty[lat == temp_data$latitude[1] & long == temp_data$longitude[1], countyname]))
    #write.table(temp_data, file = paste0(data_dir, category, "/CMPOP_", location), sep = ",", row.names = FALSE, col.names = TRUE)
    write.table(temp_data, file = paste0(data_dir, category, "/rcp45/CMPOP_", location), sep = ",", row.names = FALSE, col.names = TRUE)
    #data <- rbind(data, temp_data)
  }
}
close(conn)
#data$ClimateGroup <- as.factor(data$ClimateGroup)
#data$County <- as.factor(data$County)

