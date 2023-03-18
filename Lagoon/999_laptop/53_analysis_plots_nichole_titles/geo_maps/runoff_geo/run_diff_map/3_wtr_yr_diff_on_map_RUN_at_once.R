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

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
in_dir <- paste0(base, "runbase/")
##############################

fileN <- "wtr_yr_cum_runbase"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
dt_tb <- get_ridof_canada(dt_tb)
head(dt_tb, 2)

dt_tb <- convert_5_numeric_clusts_to_alphabet(data_tb=dt_tb)
cluster_inf <- subset(dt_tb, select=c(location, cluster))
tgt_col <- "annual_cum_runbase"
meds <- median_diff_obs_or_modeled(dt_tb, 
                                   tgt_col=tgt_col, 
                                   diff_from="1950-2005")

meds <- median_of_diff_of_medians(meds)

clusters <- unique(meds$cluster)
emissions <- c("RCP 4.5", "RCP 8.5")
future_rn_pr <- c("2026-2050", "2051-2075", "2076-2099")

#######
#######     Percentage perc_difference of medians of annual precip
#######
em <- emissions[1]; rp <- future_rn_pr[1]; clust <- clusters[1]
subtitle <- "Diff. of medians\nof cum. runoff (Water Year)"

min_diff_perc <- min(meds$perc_med_of_diffs_of_meds)
max_diff_perc <- max(meds$perc_med_of_diffs_of_meds)

for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- meds %>%
               filter(emission == em & time_period==rp) %>%
               data.table()
    # min_diff_perc <- min(curr_dt$perc_med_of_diffs_of_meds)
    # max_diff_perc <- max(curr_dt$perc_med_of_diffs_of_meds)

    title <- paste0("25-year/24-hour design storm",
                    "\n", "percent difference from historical", 
                    "\n", em, ", ", rp)

    assign(x = paste0(gsub("[.]", "", 
                      gsub("\ ", "", tolower(em))), "_",
                      gsub(pattern = "-", replacement = "_", x = rp)),
           value ={geo_map_of_diffs(dt_dt = curr_dt, 
                                    col_col = "perc_med_of_diffs_of_meds",
                                    minn=min_diff_perc, 
                                    maxx=max_diff_perc,
                                    ttl = title, 
                                    subttl= subtitle)})
  }
}
perc_diff_45 <- ggarrange(plotlist = list(rcp45_2026_2050,
                                          rcp45_2051_2075,
                                          rcp45_2076_2099),
                          ncol=3, nrow=1, common.legend=FALSE)

perc_diff_85 <- ggarrange(plotlist = list(rcp85_2026_2050,
                                          rcp85_2051_2075,
                                          rcp85_2076_2099),
                          ncol=3, nrow=1, common.legend=FALSE)

diff_plot_dir <- paste0(base, "plots/maps/runoff/diffs/")
if(dir.exists(diff_plot_dir) == F){
  dir.create(path=diff_plot_dir, recursive = T)}

ggsave(filename = paste0("perc_diff_85_12_inch_wide.png"), 
       plot=perc_diff_85, 
       width=12, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_45_12_inch_wide.png"), 
       plot=perc_diff_45, 
       width=12, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_85_14_inch_wide.png"), 
       plot=perc_diff_85, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_45_14_inch_wide.png"), 
       plot=perc_diff_45, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_85_16_inch_wide.png"), 
       plot=perc_diff_85, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_45_16_inch_wide.png"), 
       plot=perc_diff_45, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)


