rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

b <- "/Users/hn/Documents/GitHub/Ag/Lagoon/"
source_path_1 = paste0(b, "core_lagoon.R")
source_path_2 = paste0(b, "core_plot_lagoon.R")
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
##############################################################
##############################################################
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
in_dir <- paste0(base, "runbase/")
##############################################################
all_runoff <- readRDS(paste0(in_dir, "wtr_yr_cum_runbase.rds"))
all_runoff <- get_ridof_canada(all_runoff)
head(all_runoff, 2)

all_runoff <- all_runoff %>%
              filter(time_period != "2006-2025")%>%
              data.table()

all_runoff <- all_runoff %>%
              filter(time_period != "1979-2016")%>%
              data.table()

all_runoff <- within(all_runoff,
                    remove(year, month, day,
                           evap, runoff, base_flow,
                           run_p_base , precip))

all_runoff <- convert_5_numeric_clusts_to_alphabet(all_runoff)
emissions <- sort(unique(all_runoff$emission))
############################################################
####                   
####     WE DO CLUSTERS SEPARATELY SO COLORs are VISIBLE!
####
############################################################
tgt_col <- "annual_cum_runbase"
meds <- median_diff_obs_or_modeled(dt = all_runoff, 
                                   tgt_col = tgt_col, 
                                   diff_from="1950-2005")
meds <- median_of_diff_of_medians(meds)
time_ps <- sort(unique(meds$time_period))
tp <- time_ps[1]
em <- emissions[1]

unbiased_min <- min(meds$perc_med_of_diffs_of_meds)
unbiased_max <- max(meds$perc_med_of_diffs_of_meds)

for (em in emissions){
  for (tp in time_ps){
    plt_dt_unbias <- meds %>% 
                     filter(emission == em & time_period==tp) %>% 
                     data.table()
                     
    unbiased_min <- min(plt_dt_unbias$perc_med_of_diffs_of_meds)
    unbiased_max <- max(plt_dt_unbias$perc_med_of_diffs_of_meds)

    unbias_clr_lim <- c(floor(unbiased_min), ceiling(unbiased_max))

    assign(x = paste0("unbias_", 
                      gsub("[.]", "", gsub("\ ", "", tolower(em))),
                      "_", gsub("-", "_", tp)),
           value = geo_map_perc_diff(dt_dt = plt_dt_unbias,
                          col_col = "perc_med_of_diffs_of_meds", 
                                     color_limit = unbias_clr_lim) + 
                   ggtitle(label=tp))
  }
}

unbias_45 <- ggarrange(plotlist = list(unbias_rcp45_2026_2050,
                                       unbias_rcp45_2051_2075,
                                       unbias_rcp45_2076_2099),
                         ncol=3, nrow=1, common.legend=FALSE)

unbias_45_title <- paste0("Difference (%) in annual runoff ", 
                          "(RCP 4.5)")

unbias_45 <- annotate_figure(unbias_45,
                            top = text_grob(unbias_45_title, 
                                            color="red",
                                            face = "bold", 
                                            size = 14))

####### 8.5
unbias_85_title <- paste0("Difference (%) in annual runoff ", 
                          "(RCP 8.5)")


unbias_85 <- ggarrange(plotlist = list(unbias_rcp85_2026_2050,
                                       unbias_rcp85_2051_2075,
                                       unbias_rcp85_2076_2099),
                         ncol=3, nrow=1, common.legend=FALSE)

unbias_85 <- annotate_figure(unbias_85,
                            top = text_grob(unbias_85_title, 
                                            color="red",
                                            face = "bold", 
                                            size = 14))


plot_base <- paste0(base, "plots/maps/runoff/")

diff_plot_dir <- paste0(plot_base, "diffs/")
if (dir.exists(diff_plot_dir) == F) {
  dir.create(path = diff_plot_dir, recursive = T)}
print (diff_plot_dir)

ggsave(filename = paste0("unbias_45_14_inch_wide.png"), 
       plot=unbias_45, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_14_inch_wide.png"), 
       plot=unbias_85, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)



ggsave(filename = paste0("unbias_45_16_inch_wide.png"), 
       plot=unbias_45, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_16_inch_wide.png"), 
       plot=unbias_85, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

