rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

base <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/rain/"
in_dir <- paste0(base, "/03_med_diff_med_no_bias/")
plot_dir <- paste0(base, "plots/monthly/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
############################################################


fileN <- "detail_med_diff_med_month_rain"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

month_names <- c("Jan.", "Feb.", "Mar.", 
                 "Apr.", "May.", "Jun.", 
                 "Jul.", "Aug.", "Sep.", 
                 "Oct.", "Nov.", "Dec.")

box_title_base <- "diff. of medians (w/ no bias)"
box_subtitle <- "for each model median is taken over years, separately"

for (mon in 1:12){
  box_title <- paste0(box_title_base, " (", month_names[mon], ")")

  curr_dt <- dt_tb %>% filter(month==mon) %>% data.table()
  assign(x = paste0(month_names[mon], "_mag"),
        value ={ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_dt,
                                                  y_lab="magnitude of differences (mm)",
                                                  tgt_col="diff",
                                                  ttl=box_title, 
                                                  subttl=box_subtitle)+
                ggtitle(box_title, subtitle=box_subtitle)})
  assign(x = paste0(month_names[mon], "_perc"),
    value ={ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_dt,
                                              y_lab="differences (%)",
                                              tgt_col="perc_diff",
                                              ttl=box_title, 
                                              subttl=box_subtitle) +
            ggtitle(box_title, subtitle=box_subtitle)})
}

jan <- ggarrange(plotlist = list(Jan._perc, Jan._mag),
                 ncol = 1, nrow = 2,
                 common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_01_jan_diffs_rain.png",
       plot = jan, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Feb <- ggarrange(plotlist = list(Feb._perc, Feb._mag),
                 ncol = 1, nrow = 2,
                 common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_02_Feb_diffs_rain.png",
       plot = Feb, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Mar <- ggarrange(plotlist = list(Mar._perc, Mar._mag),
                 ncol = 1, nrow = 2,
                 common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_03_Mar_diffs_rain.png",
       plot = Mar, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Apr <- ggarrange(plotlist = list(Apr._perc, Apr._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_04_Apr_diffs_rain.png",
       plot = Apr, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

May <- ggarrange(plotlist = list(May._perc, May._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_05_May_diffs_rain.png",
       plot = May, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)


Jun <- ggarrange(plotlist = list(Jun._perc, Jun._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_06_Jun_diffs_rain.png",
       plot = Jun, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)


Jul <- ggarrange(plotlist = list(Jul._perc, Jul._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_07_Jul_diffs_rain.png",
       plot = Jul, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)


Aug <- ggarrange(plotlist = list(Aug._perc, Aug._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_08_Aug_diffs_rain.png",
       plot = Aug, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Sep <- ggarrange(plotlist = list(Sep._perc, Sep._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_09_Sep_diffs_rain.png",
       plot = Sep, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Oct <- ggarrange(plotlist = list(Oct._perc, Oct._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_10_Oct_diffs_rain.png",
       plot = Oct, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)

Nov <- ggarrange(plotlist = list(Nov._perc, Nov._mag),
                ncol = 1, nrow = 2,
                common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_11_Nov_diffs_rain.png",
       plot = Nov, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)


Dec <- ggarrange(plotlist = list(Dec._perc, Dec._mag),
                 ncol = 1, nrow = 2,
                 common.legend = TRUE, legend="bottom")

ggsave(filename = "no_bias_12_Dec_diffs_rain.png",
       plot = Dec, 
       width = 10, height = 6, units = "in",
       dpi=300, device = "png",
       path = plot_dir)
