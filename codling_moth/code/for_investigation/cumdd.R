#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
#filename <- paste0(data_dir, "/allData_grouped_counties.rds")
filename <- paste0(data_dir, "/allData_grouped_counties_rcp45.rds")
data <- data.table(readRDS(filename))

data <- subset(data, 
               select = c("CountyGroup","latitude","longitude",
                          "ClimateScenario","ClimateGroup","year",
                          "dayofyear","PercLarvaGen1","PercLarvaGen2",
                          "PercLarvaGen3","PercLarvaGen4"))

data <- data[, .(LarvaGen1 = mean(PercLarvaGen1), 
                LarvaGen2 = mean(PercLarvaGen2), 
                LarvaGen3 = mean(PercLarvaGen3), 
                LarvaGen4 = mean(PercLarvaGen4)), 
                by = c("CountyGroup", "latitude", "longitude", 
                       "ClimateScenario", "ClimateGroup", 
                       "year", "dayofyear")]
#saveRDS(data, "cumdd_data.rds")

#filename = paste0(data_dir, "cumdd_data.rds");
#data = data.table(readRDS(filename))
data$CountyGroup = as.character(data$CountyGroup)
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
data = melt(data, id = c("ClimateGroup", "CountyGroup", "latitude", "longitude", "ClimateScenario", "year", "dayofyear"))

plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <300], aes(x=dayofyear, y=value, fill=factor(variable))) +
  #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, 
               alpha=0.3) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.25) }, 
               fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 
  stat_summary(geom="line", 
               fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
  
  scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                     rgb(92, 160, 201, max=255), 
                     rgb(211, 91, 76, max=255), 
                     rgb(125, 7, 37, max=255))) + #c("black", "red","darkgreen","blue")) +
  scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                    rgb(92, 160, 201, max=255), 
                    rgb(211, 91, 76, max=255), 
                    rgb(125, 7, 37, max=255))) + #c("black", "red","darkgreen","blue")) +
  
  facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
  scale_x_continuous(breaks=seq(0,300,50)) +
  theme_bw() +
  geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
  geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
  
  geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
  geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
  labs(x = "Julian Day", y = "Cumulative population fraction", fill = "Larva Generation") +
  theme(
    panel.grid.major = element_line(size = 0.7),
    axis.text = element_text(face = "bold", size = 18),
    axis.title = element_text(face = "bold", size = 20),
    legend.title = element_text(face = "bold", size = 20),
    legend.text = element_text(size = 20),
    legend.position = "top",
    #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    strip.text = element_text(size = 18, face = "bold"))

#ggsave("cumdd_plot.png", plot, width = 45, height = 22, units = "cm")
ggsave("cumdd_plot_rcp45.png", plot, width = 45, height = 22, units = "cm")
#saveRDS(plot, "cumdd_plot.rds")

