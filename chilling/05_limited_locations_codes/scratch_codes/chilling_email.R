################################################
#
# pick one warm location and one cold location 
# compute HOURLY temperature data.
# for all models and scenarios for each month between 
# September and April.
# 
# find the total number of hours the hourly 
# temperature is in the following ranges
# (-inf, -2), (-2, 4), (4, 6), (6, 8), (8, 13), (13, 16), (16, inf)

# Then make the following plot for each location and month
# A panel of model_scenario combinations (like Matt's maps) 
# with the following time series plot 

# X axis: annual time series
# Y axis: box plot of hours in these ranges 
# Grouped by the 7 ranges above (so 7 trend lines in each plot)
################################################

rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)
source_path = "/Users/hn/Documents/GitHub/Kirti/Chilling/code/chill_plot_core.R"
source(source_path)

##############################
############################## Global variables
##############################
data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/limited/"
file_names = c("modeled_hist.rds", "rcp45.rds", "rcp85.rds")

needed_cols = c("Year", "Month", "Temp",
	            "Chill_season", "climateScenario", 
	            "CountyGroup", "location")

# iof = interval of interest
iof = c(c(-Inf, -2), c(-2, 4), 
        c(4, 6), c(6, 8), 
        c(8, 13), c(13, 16), 
        c(16, Inf))
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

# These are the order of months in climate calendar!
month_no = c(9, 10, 11, 12, 
	         1, 2, 3, 4, 
	         5, 6, 7, 8)
month_name = c("Jan", "Feb", "Mar", "Apr", 
	           "May", "Jun", "Jul", "Aug" ,
	           "Sept", "Oct", "Nov", "Dec")

weather_type = c("warm", "cold")
##############################
############################## Observed  data
##############################
observed = paste0(data_dir, "observed.rds")
observed = data.table(readRDS(observed))
observed = subset(observed, select=needed_cols)

observed_warm <- filter(observed, CountyGroup == "warm")
observed_cold <- filter(observed, CountyGroup == "cold")
rm(observed)
observed_warm <- observed_warm %>% 
                 mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
                 group_by(Chill_season, Year, Month, climateScenario, temp_cat) %>% 
                 summarise(no_hours = n())

observed_cold <- observed_cold %>% 
                 mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
                 group_by(Chill_season, Year, Month, climateScenario, temp_cat) %>% 
                 summarise(no_hours = n())

###########################
########################### observed - warm
###########################
weath_type = "warm"
for(month in month_no){ 
	assign(x = paste(weath_type, "observed", month_name[month], sep="_"),
		   value = { model_month_plot(filter(observed_warm, Month==month), 
		   	                          scenario_name = "observed",
		   	                          month=month_name[month])}
		   )
}
###########################
########################### observed - cold
###########################
weath_type = "cold"
for(month in month_no){ 
	assign(x = paste(weath_type, "observed", month_name[month], sep="_"),
		   value = { model_month_plot(filter(observed_warm, Month==month), 
		   	                          scenario_name = "observed",
		   	                          month=month_name[month])}
		   )
}
###################################################
##################
##################          Modeled
##################
###################################################


#################
################# Modeled Historical
#################

#################
################# Modeled Future
#################
file_names = c("modeled_hist.rds", "rcp45.rds", "rcp85.rds")

######################
######################  RCP 45
######################
file = paste0(data_dir, file_names[2])
data = data.table(readRDS(file))
data = subset(data, select=needed_cols)

carbon_type = "rcp45"
data_warm <- filter(data, CountyGroup == "warm")
data_cold <- filter(data, CountyGroup == "cold")
rm(data)

data_warm <- data_warm %>% 
                 mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
                 group_by(Chill_season, Year, Month, climateScenario, temp_cat) %>% 
                 summarise(no_hours = n())

data_cold <- data_cold %>% 
             mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
             group_by(Chill_season, Year, Month, climateScenario, temp_cat) %>% 
             summarise(no_hours = n())

########### warm
for(scenario in unique(data_warm$climateScenario)) {	
		assign(x = paste(carbon_type, "warm", gsub(pattern = "-", replacement = "_", x = scenario), sep="_"),
		       value = { model_plot_double_facet(filter(data_warm, climateScenario==scenario), 
		       	                                 scenario_name = scenario)}
		       )
}

########### cold
for(scenario in unique(data_cold$climateScenario)) {	
		assign(x = paste(carbon_type, "cold", gsub(pattern = "-", replacement = "_", x = scenario), sep="_"),
		       value = { model_plot_double_facet(filter(data_cold, climateScenario==scenario), 
		       	                                 scenario_name = scenario)}
		       )
}

plots_list = ls(pattern="rcp")

for (plotH in plots_list){
	ggsave(filename= paste0(plotH, ".png"),
		   plot=get(plotH), 
		   path="/Users/hn/Documents/GitHub/Kirti/", 
		   width=20, height=20, unit="in", dpi=400, device="png")
}

