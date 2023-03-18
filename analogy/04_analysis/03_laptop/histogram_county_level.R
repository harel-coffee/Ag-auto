######################################################################
rm(list=ls())

library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

# library(swfscMisc) has na.count(.) in it

options(digit=9)
options(digits=9)


######################################################################
####
####         functions here
####
######################################################################

produce_dt_for_hist <- function(analog_dt, novel_dt, tgt_fip){
  analog_dt <- analog_dt %>% filter(query_county == tgt_fip)
  novel_dt <- novel_dt %>% filter(query_county == tgt_fip)

  analog_dt$analog_NNs_county[is.na(analog_dt$analog_NNs_county)] <- "no_analog"
  novel_dt$novel_NNs_county[is.na(novel_dt$novel_NNs_county)] <- "not_novel"

  analog_dt <- data.table(analog_dt)
  novel_dt <- data.table(novel_dt)

  novel_dt <- novel_dt[novel_dt$novel_NNs_county != "not_novel"]
  novel_cnt <- sum(novel_dt$novel_freq)

  if (tgt_fip %in% analog_dt$analog_NNs_county){
    self_similarity_count <- analog_dt[analog_dt$analog_NNs_county==tgt_fip, 'analog_freq']$analog_freq
    } else {
     self_similarity_count <- 0
  }

  no_analog_cnt <- analog_dt[analog_dt$analog_NNs_county=="no_analog", 'analog_freq' ]$analog_freq
  non_self_simil_cnt <- sum(analog_dt$analog_freq) - no_analog_cnt - self_similarity_count

  almost_novel_cnt <-  no_analog_cnt - novel_cnt

  vvv <- c("self similarity", "similarity to other cnty.", "almost novel", "novel")
  DT = data.table(variables = vvv,
                  counts = c(self_similarity_count, non_self_simil_cnt, almost_novel_cnt, novel_cnt))
 
  DT$variables <- factor(DT$variables, order=T, levels=vvv)
  return (DT)
}

plot_hist <- function(DT, title){
  spl_plt <- ggplot(data=DT, aes(x=variables, y=counts)) +
             geom_bar(stat="identity") +
             geom_text(data=DT,
                       aes(variables, counts, label=DT$counts), 
                       position = position_dodge(width=0.9), vjust=-.8, size=4, fontface="bold") +
             ggtitle(label = title) + 
             theme(plot.title = element_text(size=16, face="bold"),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25, "cm"),
                   legend.title = element_text(face="plain", size=16),
                   legend.text = element_text(size=14),
                   legend.position = "bottom",
                   legend.key.size = unit(.65, "cm"),
                   strip.text = element_text(size=16, face="bold", color="black"),
                   axis.text.x = element_text(face="bold", size=14, color="black", angle=-10),
                   axis.text.y = element_text(face="bold", size=14, color="black"),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_blank(),
                   axis.title.y = element_text(face="bold", size=16, 
                                               margin=margin(t=0, r=10, b=0, l=0), color="black")
                    )
}

######################################################################
####
####         Set up directories
####
######################################################################

data_sub_dirs <- c("no_no_85/", "no_w_85/", "w_no_85/", "w_w_85/", 
                   "no_no_45/", "no_w_45/", "w_no_45/", "w_w_45/")

data_dir <- paste0("/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/", data_sub_dirs[1])

param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

######################################################################
####
####           global Files
####
######################################################################
local_cnty_fips <- "local_county_fips.csv"
usa_cnty_fips <- "all_us_1300_county_fips_locations.csv"
local_fip_cnty_name_map <- "17_counties_fips_unique.csv"

local_cnty_fips <- data.table(read.csv(paste0(param_dir, local_cnty_fips), header=T, sep=",", as.is=T))
usa_cnty_fips <- data.table(read.csv(paste0(param_dir, usa_cnty_fips), header=T, sep=",", as.is=T))
local_fip_cnty_name_map <- data.table(read.csv(paste0(param_dir, local_fip_cnty_name_map), 
                                               header=T, sep=",", as.is=T))

local_fips <- unique(local_cnty_fips$fips)

model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
time_periods <- c("2026_2050", "2051_2075", "2076_2095")
emissions <- c("rcp45", "rcp85")


target_fip <- local_fips[1]
model_n <- model_names[1]
time_p <- time_periods[1]
emission <- emissions[2]

for (time_p in time_periods){
  for (target_fip in local_fips){
    for(model_n in model_names){
      
      analog_file_name <- paste("analog", model_n, emission, time_p, sep="_")
      novel_file_name <- paste("novel", model_n, emission, time_p, sep="_")
      
      analog_dat <- data.table(readRDS(paste0(data_dir, analog_file_name, ".rds")))
      novel_dat <- data.table(readRDS(paste0(data_dir, novel_file_name, ".rds")))

      DT <- produce_dt_for_hist(analog_dat, novel_dat, target_fip)

      target_cnty_name <- local_fip_cnty_name_map$st_county[local_fip_cnty_name_map$fips==target_fip]
      target_cnty_name <- paste(unlist(strsplit(target_cnty_name, "_"))[2], 
                          unlist(strsplit(target_cnty_name, "_"))[1], sep= ", ")
      titlem <- paste0(target_cnty_name, 
                      " (", 
                      paste(unlist(strsplit(time_p, "_"))[1], 
                            unlist(strsplit(time_p, "_"))[2], sep="-"),
                      ", ", model_n, ")" )

      assign(x = gsub("-", "_", model_n), value = {plot_hist(DT, titlem)})
    }
    assign(x = paste0("plot_", target_fip) , 
           value={ggarrange(plotlist = list(bcc_csm1_1_m, BNU_ESM, CanESM2,
                                            CNRM_CM5, GFDL_ESM2G, GFDL_ESM2M),
                            ncol = 6, nrow = 1, common.legend = TRUE)})
  }
  assign(x = paste0("plot_", time_p) , 
           value={ggarrange(plotlist = list(plot_16027, plot_41021, plot_41027, 
                                            plot_41049, plot_41059, plot_53001, 
                                            plot_53005, plot_53007, plot_53013, 
                                            plot_53017, plot_53021, plot_53025, 
                                            plot_53037, plot_53039, plot_53047, 
                                            plot_53071, plot_53077),
                            ncol = 1, nrow = 17, common.legend = TRUE)})
}

master_path <- paste0(data_dir, "/histograms/")
if (dir.exists(master_path) == F) { dir.create(path = master_path, recursive = T)}

ggsave("plot_2026_2050.png", plot_2026_2050, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)

ggsave("plot_2051_2075.png", plot_2051_2075, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)

ggsave("plot_2076_2095.png", plot_2076_2095, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)



