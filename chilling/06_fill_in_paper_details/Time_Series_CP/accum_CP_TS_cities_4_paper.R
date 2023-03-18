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

# 3. Plotting -------------------------------------------------------------
##################################
##                              ##
##      Accumulation plots      ##
##                              ##
##################################
accum_plot <- function(data, y_name, due){
  y = eval(parse(text =paste0( "data$", y_name)))
  lab = paste0("Chill Portion accumulated by ", due)

  acc_plot <- ggplot(data = data) +
              geom_point(aes(x = year, y = y, fill = emission),
                         alpha = 0.25, shape = 21) +
              geom_smooth(aes(x = year, y = y, color = emission),
                          method = "lm", size=.8, se = F) +
              facet_wrap( ~ city) + # ~ emission
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
                    legend.position="bottom", # none
                    legend.key.size = unit(1.5, "line"),
                    legend.spacing.x = unit(.05, 'cm'),
                    legend.text=element_text(size=12),
                    panel.spacing.x =unit(.75, "cm")
                    )
  return(acc_plot)
}

# accum_hist_plot <- function(data, y_name, due){
#   y = eval(parse(text =paste0( "data$", y_name)))
#   lab = paste0("Chill units accumulated by ", due, " historically")

#   hist_plt <- ggplot(data = data) +
#               geom_point(aes(x = year, y = y), alpha = 0.4,
#                              shape = 21, fill = "#21908CFF") +
#               geom_smooth(aes(x = year, y = y), method = "lm",
#                               se = F, size=.5, color = "#21908CFF") +
#               facet_wrap( ~ city) +
#               ylab("Accum. chill units") +
#               xlab("year") +
#               scale_x_continuous(limits = c(1975, 2020)) +
#               ggtitle(label = lab,
#                       subtitle = "by location") +
#               theme(plot.title = element_text(size = 14, face="bold", color="black"),
#                     plot.subtitle = element_text(size = 12, face="plain", color="black"),
#                     axis.text.x = element_text(size = 10, face = "bold", color="black"),
#                     axis.text.y = element_text(size = 10, face = "bold", color="black"),
#                     axis.title.x = element_text(size = 12, face = "bold", color="black", 
#                                                 margin = margin(t=8, r=0, b=0, l=0)),
#                     axis.title.y = element_text(size = 12, face = "bold", color="black",
#                                                 margin = margin(t=0, r=8, b=0, l=0)),
#                     strip.text = element_text(size=14, face = "bold"),
#                     legend.margin=margin(t=.1, r=0, b=0, l=0, unit='cm'),
#                     legend.title = element_blank(),
#                     legend.position="bottom", 
#                     legend.key.size = unit(1.5, "line"),
#                     legend.spacing.x = unit(.05, 'cm'),
#                     legend.text=element_text(size=12),
#                     panel.spacing.x =unit(.75, "cm")
#                     )
#   return(hist_plt)
# }

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


for (ct in unique(summary_comp$city)){
  
  A <- summary_comp %>% filter(city == ct) %>% data.table()
    
  # summary_comp_loc_medians <- A %>%
  #                            filter(model != "observed") %>%
  #                            group_by(city, year, model, emission, time_period, chill_season) %>%
  #                            summarise_at(.funs = funs(med = median), vars(sum_A1)) %>% # vars(thresh_20:sum_A1)
  #                            data.table()

  # Data frame for historical values to be used for these figures
  # summary_comp_hist <- summary_comp %>%
  #                      filter(model == "observed") %>%
  #                      group_by(city, year) %>%
  #                      summarise_at(.funs = funs(med = median), vars(thresh_20:sum_A1))

  ############################
  ##
  ##       April plot
  ##
  ############################

  sum_A1_plot <- accum_plot(data=A, y_name="sum_A1", due="Apr. 1")
  
  ggsave(plot = sum_A1_plot, paste0("CP_accum_sept_Apr1_", ct, ".png"),
         dpi=600, path=write_dir,
         height = 4, width = 4, units = "in")


}