# 
# This is obtained by copying "chill_model_organized_TS(accum_CP)"
# The title and y-labels are wrong. When we do median and group by emission
# location, model. We are not taking median.
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
write_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/Accum_CP_Sept_Apr/"
if (dir.exists(file.path(write_dir)) == F) { dir.create(path = write_dir, recursive = T)}

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

summary_comp <- summary_comp %>% filter(time_period != "1950-2005") %>% data.table()
summary_comp <- summary_comp %>% filter(time_period != "1979-2015") %>% data.table()
# summary_comp$emission[summary_comp$time_period=="1979-2015"] <- "Observed"

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

# 3. Plotting -------------------------------------------------------------
##################################
##                              ##
##      Accumulation plots      ##
##                              ##
##################################

accum_plot_dots <- function(data, y_name, due){
  y = eval(parse(text =paste0( "data$", y_name)))
  lab = paste0("Chill Portion accumulated by ", due)
  
  tt <- theme(plot.title = element_text(size = 14, face="bold", color="black"),
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

  acc_plot <- ggplot(data = data) +
              geom_point(aes(x = year, y = y, fill = "red"),
                         alpha = 0.25, shape = 21) +
              geom_smooth(aes(x = year, y = y, color = "blue"), #  
                          method = "loess", size=.8, se = T) +
              facet_grid( . ~ emission ~  city) + # ~ emission facet_wrap
              scale_color_viridis_d(option = "plasma", begin = 0, end = .7,
                                    name = "Model scenario", 
                                    aesthetics = c("color", "fill")) +
              ylab("accumulated chill portions") +
              xlab("year") + # ggtitle(label = lab) +
              tt

  return(acc_plot)
}

accum_plot_lineModels <- function(data, y_name, due){
  y = eval(parse(text =paste0( "data$", y_name)))
  lab = paste0("Chill Portion accumulated by ", due)
  
  tt <- theme(plot.title = element_text(size = 14, face="bold", color="black"),
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

  acc_plot <- ggplot(data = data) +
              geom_smooth(aes(x = year, y = y, color = "blue"), #  
                          method = "loess", size=.8, se = T) +
              geom_line(aes(x = year, y = y, colour = model)) +
              facet_grid( . ~ emission ~  city) + # ~ emission facet_wrap
              scale_color_viridis_d(option = "plasma", begin = 0, end = .7,
                                    name = "Model scenario", 
                                    aesthetics = c("color", "fill")) +
              ylab("accumulated chill portions") +
              xlab("year") + # ggtitle(label = lab) +
              tt

  return(acc_plot)
}

##########################################################################################

for (em in unique(summary_comp$emission)){
  
  A <- summary_comp %>% filter(emission == em) %>% data.table()
  sum_A1_plot <- accum_plot_dots(data=A, y_name="sum_A1", due="Apr. 1")

  ggsave(plot = sum_A1_plot, paste0("CP_accum_sept_Apr1_em_", em,".png"),
         dpi=600, path=write_dir,
         height = 4, width = 8, units = "in")
}


summary_comp$emission <- factor(summary_comp$emission, 
                                levels=c("RCP 8.5", "RCP 4.5"),
                                order=TRUE)
 
sum_A1_plot <- accum_plot_dots(data=summary_comp, y_name="sum_A1", due="Apr. 1")

ggsave(plot = sum_A1_plot, paste0("CP_accum_sept_Apr1.png"),
       dpi=600, path=write_dir,
       height = 7, width = 10, units = "in")

ggsave(plot = sum_A1_plot, paste0("CP_accum_sept_Apr1_lowQual.png"),
     dpi=400, path=write_dir,
     height = 7, width = 10, units = "in")


#########
######### 2 rows
#########

summary_comp_85 <- summary_comp %>% filter(emission == "RCP 8.5")
summary_comp_45 <- summary_comp %>% filter(emission == "RCP 4.5")

summary_comp_85$city  <- paste0(summary_comp_85$city, " - ", summary_comp_85$emission)
summary_comp_45$city  <- paste0(summary_comp_45$city, " - ", summary_comp_45$emission)

summary_comp_85 <- within(summary_comp_85, remove(emission))
summary_comp_45 <- within(summary_comp_45, remove(emission))


ict <- c("Omak - RCP 8.5", "Yakima - RCP 8.5", "Walla Walla - RCP 8.5", "Eugene - RCP 8.5")
summary_comp_85$city <- factor(summary_comp_85$city, levels = ict, order=TRUE)


ict <- c("Omak - RCP 4.5", "Yakima - RCP 4.5", "Walla Walla - RCP 4.5", "Eugene - RCP 4.5")
summary_comp_45$city <- factor(summary_comp_45$city, levels = ict, order=TRUE)


accum_plot_dots <- function(data, y_name, due){
  y = eval(parse(text =paste0( "data$", y_name)))
  lab = paste0("Chill Portion accumulated by ", due)

  acc_plot <- ggplot(data = data) +
              geom_point(aes(x = year, y = y, fill = "red"),
                         alpha = 0.25, shape = 21) +
              geom_smooth(aes(x = year, y = y, color = "blue"), #  
                          method = "loess", size=.8, se = T) +
              facet_wrap( . ~ city) + 
              scale_color_viridis_d(option = "plasma", begin = 0, end = .7,
                                    name = "Model scenario", 
                                    aesthetics = c("color", "fill")) +
              ylab("accumulated chill portions") +
              xlab("year") +
              # ggtitle(label = lab) +
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
  return(acc_plot)
}

sum_A1_plot_45 <- accum_plot_dots(data=summary_comp_45, y_name="sum_A1", due="Apr. 1")
sum_A1_plot_85 <- accum_plot_dots(data=summary_comp_85, y_name="sum_A1", due="Apr. 1")

ggsave(plot = sum_A1_plot_45, paste0("CP_accum_sept_Apr1_em_45.png"),
       dpi=600, path=write_dir,
       height = 8, width = 8, units = "in")

ggsave(plot = sum_A1_plot_85, paste0("CP_accum_sept_Apr1_em_85.png"),
       dpi=600, path=write_dir,
       height = 8, width = 8, units = "in")


