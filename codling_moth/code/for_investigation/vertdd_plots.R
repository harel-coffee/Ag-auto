#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)


#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
filename <- paste0(data_dir, "/allData_vertdd_new.rds")
#filename <- paste0(data_dir, "/allData_vertdd_new_rcp45.rds")
data <- data.table(readRDS(filename))

data$CountyGroup = as.character(data$CountyGroup)
#data[CountyGroup == 1]$CountyGroup = 'County Group 1'
#data[CountyGroup == 2]$CountyGroup = 'County Group 2'
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

d1 = subset(data, select = c("latitude", "longitude", 
                             "CountyGroup", "ClimateGroup", 
                             "ClimateScenario", "year", 
                             "month", "day", 
                             "dayofyear", 
                             "vert_Cum_dd_F", 
                             "cripps_pink", "gala", "red_deli"))

d1 = melt(d1, id = c("latitude", "longitude", 
                     "CountyGroup", "ClimateGroup", 
                     "ClimateScenario", "year", "month", 
                     "day", "dayofyear", "vert_Cum_dd_F"))
#d1[variable == "red_deli"]$variable = "red_delicious"
d1[variable == "red_deli"]$variable = "Red Delicious"
d1[variable == "gala"]$variable = "Gala"
d1[variable == "cripps_pink"]$variable = "Cripps Pink"

p1 = ggplot(d1, aes(x=dayofyear, y=value, fill=factor(ClimateGroup))) +
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.3) +
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.7) +
  stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, size = 1)+ #, aes(color=factor(Timeframe))) + , # aes(color=factor(ClimateGroup))
  scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
  scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
  facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
  #xlim(45, 165) +
  scale_x_continuous(breaks=seq(45, 150, 10), limits = c(45, 150)) +
  theme_bw() +
  labs(x = "Julian Day", y = "Proportion Full Bloom Completed", fill = "Climate Group") +
  theme(
    panel.grid.major = element_line(size = 0.7),
    axis.text = element_text(face = "bold", size = 18),
    axis.title = element_text(face = "bold", size = 20),
    legend.title = element_text(face = "bold", size = 20),
    legend.text = element_text(size = 20),
    legend.position = "top",
    #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    strip.text = element_text(size = 18, face = "bold"))

ggsave("bloom1.png", p1, width = 45, height = 22, units = "cm")
#ggsave("bloom1_rcp45.png", p1, width = 45, height = 22, units = "cm")

#plot2 = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
#  #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
#  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
#  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
#  stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
#  scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
#  scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
#  
#  facet_grid(CountyGroup ~ ClimateGroup ~ ., scales = "fixed") +
#  scale_x_continuous(breaks=seq(0,370,50)) +
#  scale_y_continuous(breaks=seq(0,5000,1000)) +
#  theme_bw() +
#  #geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
#  #geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
#  #geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
#  #geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
#  labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
#  theme(
#    panel.grid.major = element_line(size = 0.7),
#    axis.text = element_text(face = "bold", size = 18),
#    axis.title = element_text(face = "bold", size = 20),
#    legend.title = element_text(face = "bold", size = 20),
#    legend.text = element_text(size = 20),
#    legend.position = "top",
#    #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
#    strip.text = element_text(size = 18, face = "bold"))
#
#ggsave("cumdd_plot3.png", plot2, width = 37, height = 50, units = "cm")

