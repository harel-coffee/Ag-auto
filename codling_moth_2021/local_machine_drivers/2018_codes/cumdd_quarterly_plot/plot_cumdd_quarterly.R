.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(chillR)
library(tidyverse)
library(lubridate)

library(data.table)
library(ggplot2)
library(dplyr)

input_dir = "/Users/hn/Documents/01_research_data/my_aeolus_2015/all_local/4_cumdd/"
version = c("rcp45", "rcp85")

for (vers in version){
  data = readRDS(paste0(input_dir, "cumdd_CMPOP_", vers, ".rds"))
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

  #df <- data.frame(data)
  #df <- (df %>% group_by(CountyGroup, ClimateGroup, season))
  #medians <- (df %>% summarise(med = median(CumDDinF)))
  #medians <- medians$med
  #rm(df)

  data = melt(data, id = c("ClimateGroup", "CountyGroup", "season"))
  data = within(data, remove(variable))

  bplot <- ggplot(data = data, aes(x=season, y=value), group = season) + 
           geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                        position=position_dodge(width=0.5)) + 
           scale_y_continuous(limits = c(0, 6000), breaks = seq(0, 6000, by = 1000)) + 
           facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
           labs(x="", y="Cumulative degree day", color = "Climate Group") + 
           theme_bw() +
           theme(plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm"),
                 legend.position="bottom", 
                 legend.margin=margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 legend.text = element_text(size=18, face="plain"),
                 legend.key.size = unit(1, "cm"), 
                 legend.spacing.x = unit(0.5, 'cm'),
                 panel.grid.major = element_line(size = 0.1),
                 panel.grid.minor = element_line(size = 0.1),
                 strip.text = element_text(size= 20, face = "plain"),
                 axis.text = element_text(face = "plain", size = 10, color="black"),
                 axis.title.x = element_text(face = "plain", size = 10, 
                                             margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.x = element_text(size = 18, color="black"), # tick text font size
                 axis.text.y = element_text(size = 18, color="black"),
                 axis.title.y = element_text(face = "plain", size = 22, 
                                             margin = margin(t=0, r=7, b=0, l=0))
               )

  out_name = paste0("cumdd_qrt_", vers, ".png")
  output_dir = input_dir
  ggsave(out_name, bplot, width=7.5, height=3.5, unit="in", path=output_dir, dpi=300, device="png")
}


rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

input_dir = "/Users/hn/Documents/01_research_data/my_aeolus_2015/all_local/4_cumdd/"
version = c("rcp45", "rcp85")

for (vers in version){
  data = readRDS(paste0(input_dir, "cumdd_CMPOP_", vers, ".rds"))
  data = subset(data, select = c("ClimateGroup", "CountyGroup",
                                 "dayofyear", "CumDDinF"))
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

  #df <- data.frame(data)
  #df <- (df %>% group_by(CountyGroup, ClimateGroup, season))
  #medians <- (df %>% summarise(med = median(CumDDinF)))
  #medians <- medians$med
  #rm(df)

  data = melt(data, id = c("ClimateGroup", "CountyGroup", "season"))
  data = within(data, remove(variable))

  bplot <- ggplot(data = data, aes(x=season, y=value), group = season) + 
           geom_boxplot(outlier.size=-.3, notch=FALSE, width=.4, lwd=.15, aes(fill=ClimateGroup), 
                        position=position_dodge(width=0.5)) + 
           scale_y_continuous(limits = c(0, 6000), breaks = seq(0, 6000, by = 1000)) + 
           facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
           labs(x="", y="Cumulative degree day", color = "Climate Group") + 
           theme_bw() +
           theme(legend.position="bottom", 
                 legend.margin = margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 legend.text = element_text(size=8, face="plain"),
                 legend.key.size = unit(.4, "cm"), 
                 legend.spacing.x = unit(0.2, 'cm'),
                 panel.grid.major = element_line(size = 0.1),
                 panel.grid.minor = element_line(size = 0.1),
                 strip.text = element_text(size = 7, face = "plain"),
                 axis.title.x = element_text(size = 10, face = "plain", margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.x = element_text(size = 7, color="black"), # tick text font size
                 axis.text.y = element_text(size = 7, color="black"),
                 axis.title.y= element_text(size = 9, face = "plain", margin = margin(t=0, r=7, b=0, l=0)),
                 plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
                 )

  out_name = paste0("cumdd_qrt_", vers, ".png")
  output_dir = input_dir
  ggsave(out_name, bplot, width=7.5, height=3.5, unit="in", path=output_dir, dpi=300, device="png")
}



