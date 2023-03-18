# """
# This driver will comupte the percentage diff between future CPs and observed CPs.
# and generate a map of colors for them.
# 
# """
rm(list=ls())

library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)


core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)


data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)

remove_montana <- function(data_dt, LocationGroups_NoMontana){
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$lat, "_", data_dt$long)
  }
  data_dt <- data_dt %>% filter(location %in% LocationGroups_NoMontana$location)
  return(data_dt)
}


sept_summary_comp <- readRDS(paste0(data_dir, "sept_summary_comp.rds")) %>%
                     data.table()

head(sept_summary_comp, 2)
dim(sept_summary_comp)

keep_cols <- c("location", "sum_A1", "model", "emission", "time_period")

sept_summary_comp <- subset(sept_summary_comp, select=keep_cols)

sept_summary_comp <- sept_summary_comp %>% 
                     filter(!(time_period %in% c("2006-2025", "1950-2005"))) %>%
                     data.table()

sept_summary_comp <- remove_montana(sept_summary_comp, LocationGroups_NoMontana)


###### Change time period for sake of plotting:
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2025_2050"] = "2025-2050"
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2051_2075"] = "2051-2075"
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2076_2100"] = "2076-2099"

sept_summary_comp$time_period[sept_summary_comp$model == "observed"] <- "Observed"

sept_summary_comp$emission[sept_summary_comp$emission == "rcp45"] = "RCP 4.5"
sept_summary_comp$emission[sept_summary_comp$emission == "rcp85"] = "RCP 8.5"
sept_summary_comp$emission[sept_summary_comp$emission == "historical"] = "Observed"


time_periods = c("Observed", "2026-2050", "2051-2075", "2076-2099")
sept_summary_comp$time_period = factor(sept_summary_comp$time_period, levels = time_periods, order=T)

sept_summary_comp_yearsMedian_perModel <- sept_summary_comp %>%
                                          group_by(location, model, emission, time_period) %>%
                                          summarise(CP_median_A1=median(sum_A1)) %>%
                                          data.table()
 
diffs <- projec_diff_from_hist(sept_summary_comp_yearsMedian_perModel)

diffs_median <- diffs %>% 
                group_by(location, time_period, emission) %>%
                summarise(CP_diff_median = median(perc_diff)) %>%
                data.table()

diffs_median$CP_diff_median[diffs_median$CP_diff_median < -40] = -40

plot_base <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/03_Springer_3/")
# plot_base <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/"
if (dir.exists(plot_base) == F) {dir.create(path = plot_base, recursive = T)}

core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)

diffs_median$emission <- factor(diffs_median$emission, 
                                levels=c("RCP 8.5", "RCP 4.5"),
                                order=TRUE)


diff_CP_map_w_legend <- function(data, color_col) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  if (color_col=="CP_diff_median"){
     low_lim = min(data$CP_diff_median)
     up_lim = max(data$CP_diff_median)
    } else if (color_col=="median_over_model"){
     low_lim = min(data$median_over_model)
     up_lim = max(data$median_over_model)
  }
  x <- data$location
  x <- sapply(x, 
             function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  data$lat <- as.numeric(lat)
  data$long <- as.numeric(long)
  data <- data.table(data)

  legend_d <- "CP accumulation difference (%) \n"

  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           labs(x = "longitude (degree)", y = "latitude (degree)") +
           scale_y_continuous(breaks = seq(40, 50, by = 5)) + 
           scale_x_continuous(breaks = seq(-125, -110, by = 10)) + 
           theme(axis.title.y = element_text(color = "black", face="bold"),
                 axis.title.x = element_text(color = "black", face="bold"),
                 # axis.ticks.y = element_blank(), 
                 # axis.ticks.x = element_blank(),
                 axis.text.x = element_text(color = "black"),
                 axis.text.y = element_text(color = "black"),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.text=element_text(size=12), #, face="bold"
                 legend.title=element_text(size=15),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=0.03, b=0.03, r=0.2, l=0.2, unit = 'cm')
                 ) + 
           labs(fill=guide_legend(title=legend_d)) + 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim),
                                 breaks = c(-200, -75, -40, -25, 0, 25, 40, 75, 200),
                                 name=legend_d)
           
}
a_map <- diff_CP_map_w_legend(data = diffs_median, color_col = "CP_diff_median")
a_map

qual = 400
W = 7.5
H = 5.5
ggsave(filename = paste0("CP_diff_perc_Sept_Apr1_centered.pdf"), 
       plot=a_map, 
       width=W, height=H, units="in", 
       dpi=qual, device="pdf", path=plot_base)


diffs_median_85 <- diffs_median %>% filter(emission=="RCP 8.5")

core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)
rcp85_map <- diff_CP_map_one_emission(data = diffs_median_85, color_col = "CP_diff_median")

W = 7.5
H = 3.3

ggsave(filename = paste0("CP_diff_perc_Sept_Apr1st_centered_85.pdf"), 
       plot=rcp85_map, 
       width=W, height=H, units="in", 
       dpi=qual, device="pdf", 
       path=plot_base)



