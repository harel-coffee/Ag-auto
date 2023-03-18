#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#############################################################################################
##
## This file plots the plots in the "Regional Plots" of the website
## associated with Degree Days
##
#############################################################################################
model = "rcp45"

if (model=='rcp45'){
	fileName = "/allData_grouped_counties_rcp45.rds"
}else{
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

plot1 = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
	#geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
	stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
	stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
	stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 	 
	scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
	scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
	 
	facet_grid(. ~ CountyGroup, scales="free") +
	scale_x_continuous(breaks=seq(0,370,25)) +
	scale_y_continuous(breaks=seq(0,4500,250)) +
	theme_bw() +
    #geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
    #geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
    #geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
    #geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
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
ggsave(plot_name, plot1, width = 45, height = 22, units = "cm")