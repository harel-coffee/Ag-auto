.libPaths("/data/hydro/R_libs35")
.libPaths()
library(tidyverse)
library(lubridate)
library(ggpubr)

library(data.table)
library(ggplot2)
library(dplyr)

start_time <- Sys.time()

input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/4_cumdd/"
input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/"
version = c("rcp45", "rcp85")

make_non_overlapping <- function(overlap_dt){
  old_time_periods <- c("2040's", "2060's", "2080's", "Historical")
  new_time_periods <- c("2026-2050", "2051-2075", "2076-2095", "Historical")

  ########## Historical
  hist <- overlap_dt %>% filter(ClimateGroup == "Historical")

  ########## 2040
  F1 <- overlap_dt %>% filter(ClimateGroup == "2040's")
  F1 <- F1 %>% filter(year>=2025 & year<=2050)
  F1$ClimateGroup <- new_time_periods[1]

  ########## 2060
  F2 <- overlap_dt %>% filter(ClimateGroup == "2060's")
  F2 <- F2 %>% filter(year>=2051 & year<=2075)
  F2$ClimateGroup <- new_time_periods[2]

  ########## 2080
  F3 <- overlap_dt %>% filter(ClimateGroup == "2080's")
  F3 <- F3 %>% filter(year>=2076)
  F3$ClimateGroup <- new_time_periods[3]

  non_overlap_dt <- rbind(hist, F1, F2, F3)
  non_overlap_dt$ClimateGroup <- factor(non_overlap_dt$ClimateGroup, levels=new_time_periods, order=T)
}

cleanex <- function(data){
  data = subset(data, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumDDinF"))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  # add the new season column
  data[, season := as.character(0)]
  data[data[ , data$dayofyear <= 90]]$season = "QTR 1"
  data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "QTR 2"
  data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "QTR 3"
  data[data[ , data$dayofyear >= 274]]$season = "QTR 4"
  data = within(data, remove(dayofyear))
  data$season = factor(data$season, levels = c("QTR 1", "QTR 2", "QTR 3", "QTR 4"))
  return(data)
}

data_45 = readRDS("./cumdd_CMPOP_rcp45.rds")
data_85 = readRDS("./cumdd_CMPOP_rcp85.rds")

data_45_clean <- cleanex(data_45)
data_85_clean <- cleanex(data_85)

data_45_clean$emission <- "RCP 4.5"
data_85_clean$emission <- "RCP 8.5"

all_data <- rbind(data_45_clean, data_85_clean)

rm(data_45, data_85, data_45_clean, data_85_clean)

all_data <- melt(all_data, id = c("ClimateGroup", "CountyGroup", "season", "emission"))
# all_data <- within(all_data, remove(variable))
# all_data[, variable := NULL] # perhaps this is more efficient than the line above.

# perhaps this is more efficient than the line above
all_data <- subset( all_data, select = -c(variable) ) 

the_theme <- theme_bw() +
               theme(legend.position="bottom", 
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
                                                 margin = margin(t=0, r=7, b=0, l=0)),
                     plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
                    )

bplot <- ggplot(data = all_data, aes(x=season, y=value), group = season) + 
         geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                      position=position_dodge(width=0.5)) + 
         scale_y_continuous(limits = c(0, 6000), breaks = seq(0, 6000, by = 1000)) + 
         facet_grid(. ~CountyGroup ~ emission, scales="free", ncol=6, dir="v") + 
         labs(x="", y="cumulative degree day", color = "Climate Group") + 
         the_theme

ggsave("cumdd_qrt.png", bplot, width=15, height=7, unit="in", path="./", dpi=300, device="png")

print( Sys.time() - start_time)












