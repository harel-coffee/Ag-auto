# 
# Never Ending Project 2.
#
# Oct. 13, 2021
#

# 1. Load packages --------------------------------------------------------
rm(list=ls())
library(ggpubr) # library(plyr)
library(tidyverse)
library(data.table)
library(ggplot2)
options(digits=9)
options(digit=9)

##############################################################################
############# 
#############              ********** start from here **********
#############
##############################################################################
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
limited_locs <- read.csv(paste0(param_dir, "limited_locations.csv"), 
                                header=T, sep=",", as.is=T)

main_in_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/02/"

summary_comp <- data.table(readRDS(paste0(main_in_dir, "sept_summary_comp.rds")))
summary_comp$location <- paste0(summary_comp$lat, "_", summary_comp$long)
limited_locs$location <- paste0(limited_locs$lat, "_", limited_locs$long)
summary_comp <- summary_comp %>% filter(location %in% limited_locs$location)
summary_comp <- left_join(summary_comp, limited_locs)

######################################################################
#####
#####                Clean data
#####
#######################################################################
# summary_comp <- summary_comp %>% filter(emission != "rcp85") %>% data.table()
summary_comp$emission[summary_comp$emission=="rcp45"] <- "RCP 4.5"
summary_comp$emission[summary_comp$emission=="rcp85"] <- "RCP 8.5"

unique(summary_comp$emission)
unique(summary_comp$time_period)

summary_comp <- within(summary_comp, remove(location, lat, long, thresh_20, 
                                            thresh_25, thresh_30, thresh_35,
                                            thresh_40, thresh_45, thresh_50,
                                            thresh_55, thresh_60, thresh_65,
                                            thresh_70, thresh_75, sum_J1, sum_F1, sum_M1,
                                            sum))

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

summary_comp <- summary_comp %>% 
                filter(city %in% ict) %>% 
                data.table()

summary_comp$city <- factor(summary_comp$city, levels = ict, order=TRUE)

observed <- summary_comp %>%
            filter(time_period == "1979-2015")%>% 
            data.table()

MH <- summary_comp %>% 
      filter(time_period == "1950-2005")%>% 
      data.table()

F85 <- summary_comp %>% 
       filter(emission == "RCP 8.5")%>% 
       data.table()

F45 <- summary_comp %>% 
       filter(emission == "RCP 4.5")%>% 
       data.table()

obs = data.frame()
for (a_model in unique(F45$model)){
  A <- copy(observed)
  A$model <- a_model
  obs <- rbind(obs, A)
}
obs <- data.table(obs)
# 3. Plotting -------------------------------------------------------------
##################################
##                              ##
##      Accumulation plots      ##
##                              ##
##################################

param_dr <- paste0("/Users/hn/Documents/00_GitHub/Ag/Bloom/parameters/")
chill_doy_map <- read.csv(paste0(param_dr, "/chill_DoY_map.csv"), as.is=TRUE)
cls <- "red" # deepskyblue, darkgreen, orange, grey40

facet_accum_CP_TS_Cloud_SpringerSub2 <- function(d1, colname="sum_A1", due="Apr. 1", fil="Apr. 1"){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]

  ggplot(d1, aes(x=year, y=get(colname), fill=emission, group=emission, colour=emission)) +
  labs(x = "chill year", y = "accumulated chill portions") +
  facet_grid( . ~ city ~ model) +
  geom_line() +
  scale_color_manual(values=c("#CC6666", "#9999CC")) + 
  scale_fill_manual(values=c("#CC6666", "#9999CC")) + 
  theme(plot.title = element_text(size = 14, face="bold", color="black"),
        plot.subtitle = element_text(size = 12, face="plain", color="black"),
        panel.grid.major = element_line(size=0.1),
        panel.grid.minor = element_line(size=0.1),
        axis.text.x = element_text(size = 10, color="black"),
        axis.text.y = element_text(size = 10, color="black"),
        axis.title.x = element_text(size = 12, face = "bold", color="black", 
        margin = margin(t=8, r=0, b=0, l=0)),
        axis.title.y = element_text(size = 12, face = "bold", color="black",
                                          margin = margin(t=0, r=8, b=0, l=0)),
        strip.text = element_text(size=14, face = "bold"),
        legend.margin=margin(t=.1, r=0, b=0, l=0, unit='cm'),
        legend.title = element_blank(),
        legend.position="none", # 
        legend.key.size = unit(1.5, "line"),
        legend.spacing.x = unit(.05, 'cm'),
        legend.text=element_text(size=12),
        panel.spacing.x =unit(.75, "cm")
        )

}

##########################################################################################
##########################################################################################
#######
#######   RCP 8.5
#######
##########################################################################################
write_dir <- "/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/02_Springer_2/figure_202_ObsVsModHist_SepMods/"
if (dir.exists(file.path(write_dir)) == F) { dir.create(path = write_dir, recursive = T)}

pwidth = 41
pheight = 12


obs$emission <- "Observed"
MH$emission <- "RCP 8.5"
d_85 <- rbind(MH, F85, obs)
d_85$emission <- factor(d_85$emission, levels = c("RCP 8.5", "Observed"), order=TRUE)

ggsave(plot = plottt, paste0("RCP85.pdf"),
       dpi=400, path=write_dir,
       height = pheight, width = pwidth, units = "in")




obs$emission <- "Observed"
MH$emission <- "RCP 4.5"
d_45 <- rbind(MH, F45, obs)
d_45$emission <- factor(d_45$emission, levels = c("RCP 4.5", "Observed"), order=TRUE)

ggsave(plot = plottt, paste0("RCP45.pdf"),
       dpi=400, path=write_dir,
       height = pheight, width = pwidth, units = "in")


