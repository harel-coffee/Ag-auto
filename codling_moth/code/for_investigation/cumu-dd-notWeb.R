#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#############################################################################################
##
## This file plots the plots 
## that are not on the website. However, something similar 
## to the left of the image cumdd_plot3_rcp45.png is produced.
## With a little bit of change, perhaps, we can produce something
## similar to Regional Plots, Degree Days.
##
#############################################################################################
model = "rcp45"

if (model=='rcp45'){
	fileName = "/allData_grouped_counties_rcp45.rds"
}else if (model=='rcp85'){
    fileName = "/allData_grouped_counties.rds"
}

##
## Read Data
##
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
filename <- paste0(data_dir, fileName)
data <- data.table(readRDS(filename))

##
## e.g. on my computer  
## allData_one_location_rcp45 is found at : /data/hydro/users/giridhar/giridhar/codmoth_pop on Aeolus
## 
# data = readRDS("/Users/hn/Documents/GitHub/Kirti/Giridhar/R/Aeolus-Data/allData_one_location_rcp45.rds")
data$CountyGroup = as.character(data$CountyGroup)
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
data = subset(data, select = c("ClimateGroup", "CountyGroup", "latitude", "longitude", "ClimateScenario", "year", "dayofyear", "CumDDinF"))
data$CumDD = data$CumDDinF

data <- data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", "ClimateScenario", "ClimateGroup", "dayofyear")]
data <- data[, .(CumDD = mean(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", "ClimateScenario", "ClimateGroup", "dayofyear")]

plot2 = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
	#geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
	stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
	stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
	stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
	scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
	scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
	
	facet_grid(CountyGroup ~ ClimateGroup ~ ., scales = "fixed") +
	scale_x_continuous(breaks=seq(0,370,50)) +
	scale_y_continuous(breaks=seq(0,5000,1000)) +
	theme_bw() +
    labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      axis.text = element_text(face = "bold", size = 18),
      axis.title = element_text(face = "bold", size = 20),
      legend.title = element_text(face = "bold", size = 20),
      legend.text = element_text(size = 20),
      legend.position = "top",
      strip.text = element_text(size = 18, face = "bold"))

if (model=='rcp45'){
	plot_name = "cumdd_plot2.png"
}else{
    plot_name = "cumdd_plot2_rcp45.png"}
ggsave(plot_name, plot2, width = 45, height = 22, units = "cm")