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

in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/runbase/"
plot_dir <- paste0(in_dir, "narrowed_runbase/geo/")
if (dir.exists(plot_dir)==F){
  dir.create(path=plot_dir, recursive = T)}
print (plot_dir)

##############################
fileN <- "chunk_cum_runbase"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

tgt_col <- "chunk_cum_runbase"
meds <- compute_median_diff(dt_tb, tgt_col=tgt_col)
meds$perc_diff <- meds$perc_diff / 100

cut_borders <- c(-Inf, -0.50, -0.20, -0.15, -0.1, -0.05, 
                 0, 
                 0.05, 0.1, 0.15, 0.20, 0.50, Inf)
meds$cutss <- cut(meds$perc_diff, cut_borders)

emissions <- c("RCP 4.5", "RCP 8.5")
future_rn_pr <- c("2026-2050", "2051-2075", "2076-2099")

#######
#######     Percentage perc_difference of medians of annual precip
#######
subtitle <- "Diff. of medians of cum. runoff (Sept.-Mar., in percentage)"
em <- emissions[1]
rp <- future_rn_pr[1]
for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- meds %>%
               filter(emission == em & time_period==rp) %>%
               data.table()
    title <- paste0(em, " (", rp, ")")
    
    assign(x = paste0(gsub(pattern = " ", 
                           replacement = "_", 
                           x = em),
                      "_",
                      gsub(pattern = "-", 
                           replacement = "_", 
                           x = rp)),
           value ={geo_map_of_diffs_discrete_cuts(dt = curr_dt, 
                                                  col_col = "cutss" , 
                                                  ttl = title, 
                                                  subttl= subtitle)})

  }
}

perc_diff_figs <- ggarrange(plotlist = list(RCP_8.5_2026_2050,
                                            RCP_8.5_2051_2075,
                                            RCP_8.5_2076_2099,
                                            RCP_4.5_2026_2050,
                                            RCP_4.5_2051_2075,
                                            RCP_4.5_2076_2099),
                           ncol = 3, nrow = 2,
                           common.legend = TRUE)

ggsave(filename = "run_perc_diff_medians_Sept_Mar_dis.png", 
       plot = perc_diff_figs, 
       width = 10, height = 7, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

rm(RCP_4.5_2026_2050, RCP_8.5_2026_2050,
   RCP_4.5_2051_2075, RCP_8.5_2051_2075,
   RCP_4.5_2076_2099, RCP_8.5_2076_2099, perc_diff_figs)



