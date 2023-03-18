#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
#filename <- paste0(data_dir, "/allData_revised.rds")
#filename <- paste0(data_dir, "/allData.rds")
filename <- paste0(data_dir, "/allData_rcp45.rds")
filename <- paste0(data_dir, "/allData_grouped_counties_rcp45.rds")

data <- data.table(readRDS(filename))

#data = data[, .(PercAdultGen1 = median(PercAdultGen1), 
#                PercAdultGen2 = median(PercAdultGen2), 
#                PercAdultGen3 = median(PercAdultGen3), 
#                PercAdultGen4 = median(PercAdultGen4), 
#                PercLarvaGen1 = median(PercLarvaGen1), 
#                PercLarvaGen2 = median(PercLarvaGen2), 
#                PercLarvaGen3 = median(PercLarvaGen3), 
#                PercLarvaGen4 = median(PercLarvaGen4), 
#                CumDDinC = median(CumDDinC), 
#                CumDDinF = median(CumDDinF)), 
#                by = c("ClimateGroup", "year", "month", "day", "dayofyear")]

#data <- subset(data, select = c("ClimateGroup", "month", 
#		"PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4", 
#		"PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))

#data_melted = melt(data, c("ClimateGroup", "month"), 
#                   c("PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4", 
#                     "PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"), 
#                   variable.name = "Generations")

#p = ggplot(data = data_melted, aes(x = Generations, y = value, fill = ClimateGroup)) +
#  geom_boxplot() + coord_flip() +
#  facet_wrap(~month)

#saveRDS(p, paste0(data_dir, "/", "popplot.rds"))

#Chelan, Douglas, Okanogan
#Grant, Adams
#Umatilla, Walla Walla
#data[County == "Grant" | County == "Adams", County := "Grant_Adams"]
#data[County == "Chelan" | County == "Douglas" | County == "Okanogan", County := "Chelan_Douglas_Okanogan"]
#data[County == "Umatilla" | County == "Walla Walla", County := "Umatilla_WallaWalla"]

#data = data[, .SD[.N], by=c("ClimateGroup", "ClimateScenario", "latitude", "longitude", "County", "year", "month")]
#saveRDS(data, paste0(data_dir, "/", "subData_rcp45.rds"))

#data = subset(data, select = c("ClimateGroup", "ClimateScenario", "latitude", "longitude", "County", "year", "month", "DailyDD"))
#data$MonthGroup = 0
#data[month == 1 | month == 2 | month == 3, MonthGroup := 3]
#data[month == 4 | month == 5 | month == 6, MonthGroup := 6]
#data[month == 7 | month == 8 | month == 9, MonthGroup := 9]
#data[month == 10 | month == 11 | month == 12, MonthGroup := 12]
#data = data[, .(monthlyDD = sum(DailyDD)), by = c("ClimateGroup", "ClimateScenario", "latitude", "longitude", "County", "year", "MonthGroup")]
#data = data[, .(monthlyDD = sum(DailyDD)), by = c("ClimateGroup", "ClimateScenario", "latitude", "longitude", "County", "year", "month")]
#saveRDS(data, paste0(data_dir, "/", "subDDData_rcp45.rds"))
#saveRDS(data, paste0(data_dir, "/", "subDDData_month_groups.rds"))
#saveRDS(data, paste0(data_dir, "/", "subDDData_month_groups_rcp45.rds"))

data = data[latitude == 46.34375 & longitude == -119.21875]
#data = data[latitude == 48.96875 & longitude == -119.46875]
saveRDS(data, paste0(data_dir, "/", "allData_one_location_rcp45.rds"))
