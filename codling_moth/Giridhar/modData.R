#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
#filename <- paste0(data_dir, "/allData.rds")
filename <- paste0(data_dir, "/allData_grouped_counties_rcp45.rds")
data <- data.table(readRDS(filename))
#data = data[, .(PercAdultGen1 = median(PercAdultGen1), PercAdultGen2 = median(PercAdultGen2), PercAdultGen3 = median(PercAdultGen3), PercAdultGen4 = median(PercAdultGen4), PercLarvaGen1 = median(PercLarvaGen1), PercLarvaGen2 = median(PercLarvaGen2), PercLarvaGen3 = median(PercLarvaGen3), PercLarvaGen4 = median(PercLarvaGen4), CumDDinC = median(CumDDinC), CumDDinF = median(CumDDinF)), by = c("ClimateGroup", "year", "month", "day", "dayofyear")]
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
#data[County == "Grant" | County == "Adams", County := "Grant, Adams"]
#data[County == "Chelan" | County == "Douglas" | County == "Okanogan", County := "Chelan, Douglas, Okanogan"]
#data[County == "Umatilla" | County == "Walla Walla", County := "Umatilla, WallaWalla"]
#data$County = factor(data$County)
#print(levels(data$County))

loc_grp = data.table(read.csv("LocationGroups2.csv"))
loc_grp$latitude = as.numeric(loc_grp$latitude)
loc_grp$longitude = as.numeric(loc_grp$longitude)

data$CountyGroup = 0L

for(i in 1:nrow(loc_grp)) {
	data[latitude == loc_grp[i, latitude] & longitude == loc_grp[i, longitude], ]$CountyGroup = loc_grp[i, locationGroup]
}

saveRDS(data, paste0(data_dir, "/", "allData_grouped_counties_rcp45.rds"))
