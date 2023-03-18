rm(list=ls())
library(data.table)
library(dplyr)

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/snow_correctness/"
out_dir <- paste0(in_dir, "/five_locations/")

chosen <- c("47.84375_-122.34375", "48.15625_-122.34375", 
            "47.90625_-121.84375", "48.40625_-122.53125", 
            "48.40625_-122.46875")

obs <- data.table(readRDS(paste0(in_dir, "chosen_obs.rds")))
modeled <- data.table(readRDS(paste0(in_dir, "chosen_modeled.rds")))

############################################################

obs <- obs %>% filter(location == "47.84375_-122.34375")
modeled <- modeled %>% filter(location == "47.84375_-122.34375")
out_dir <- paste0(in_dir, "/one_location/")
############################################################
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}

modeled <- modeled %>% filter(time_period != "2006-2025")
modeled_F1 <- modeled %>% filter(time_period == "2026-2050")
modeled_F2 <- modeled %>% filter(time_period == "2051-2075")
modeled_F3 <- modeled %>% filter(time_period == "2076-2099")
all_together <- rbind(obs, modeled) %>% data.table()

#############################################################################
rain_portion <- function(dt_dt){
  # dt_dt$rain_portion <- apply(dt_dt[, "tmean"], MARGIN=1, FUN=rain_p)
  # dt_dt$rain_portion <- unlist(lapply(data$tmean, FUN=rain_portion))

  dt_dt <- dt_dt %>%
           mutate(rain_portion = case_when(tmean <= 0.6 ~ 0,
                                           (tmean < 3.6 & tmean> 0.6) ~ ((tmean/3) - 0.2),
                                            tmean >= 3.6 ~ 1)) %>%
           data.table()
  return(dt_dt)
}

obs <- rain_portion(obs)
modeled_F1 <- rain_portion(modeled_F1)
modeled_F2 <- rain_portion(modeled_F2)
modeled_F3 <- rain_portion(modeled_F3)
all_together <- rain_portion(all_together)

###########################################################################
write.table(obs, 
            file = paste0(out_dir, "00_obs_rain_portion.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")
write.table(modeled_F1, 
            file = paste0(out_dir, "00_F1_rain_portion.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2, 
            file = paste0(out_dir, "00_F2_rain_portion.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3, 
            file = paste0(out_dir, "00_F3_rain_portion.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together, 
            file = paste0(out_dir, "00_F_rain_portion.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

#############################################################################
annual_cum_rain <- function(data_tb){
  ##############################################################
  # input: data_tb                                             #
  #                                                            #
  # output:                                                    #
  #                                                            #
  #                                                            #
  ##############################################################
  data_tb$rain <- data_tb$precip * data_tb$rain_portion
  data_tb$snow <- data_tb$precip * (1 - data_tb$rain_portion)
  
  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_precip = cumsum(precip)) %>%
             data.table()
  
  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_rain = cumsum(rain)) %>%
             data.table()

  data_tb <- data_tb %>%
             group_by(location, year, model, emission, time_period) %>%
             mutate(annual_cum_snow = cumsum(snow)) %>%
             data.table()

  return (data_tb)
}

obs <- annual_cum_rain(obs)
modeled_F1 <- annual_cum_rain(modeled_F1)
modeled_F2 <- annual_cum_rain(modeled_F2)
modeled_F3 <- annual_cum_rain(modeled_F3)
all_together <- annual_cum_rain(all_together)

col_order <- c("location", "year", "month", 
               "day", "time_period", "cluster", "emission", "model",
               "tmean", "precip", "rain_portion", "rain", "snow",
               "annual_cum_precip", "annual_cum_rain", "annual_cum_snow")

setcolorder(obs, col_order); 
setcolorder(modeled_F1, col_order); 
setcolorder(modeled_F2, col_order); 
setcolorder(modeled_F3, col_order); 
setcolorder(all_together, col_order)
###########################################################################
###########################################################################
#
# Test if having three future time periods will change anything
all_together_1 <- all_together %>% filter(time_period == "2026-2050") %>% data.table()
all_together_2 <- all_together %>% filter(time_period == "2051-2075") %>% data.table()
all_together_3 <- all_together %>% filter(time_period == "2076-2099") %>% data.table()

col_order_1 <- colnames(modeled_F1)
setcolorder(modeled_F1, col_order_1); 
setcolorder(modeled_F2, col_order_1); 
setcolorder(modeled_F3, col_order_1); 
setcolorder(all_together_1, col_order_1); 
setcolorder(all_together_2, col_order_1); 
setcolorder(all_together_3, col_order_1); 

all.equal(all_together_1, modeled_F1)
all.equal(all_together_2, modeled_F2)
all.equal(all_together_3, modeled_F3)
######################################################################
write.table(obs, 
            file = paste0(out_dir, "01_obs_rain_cum.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F1, 
            file = paste0(out_dir, "01_F1_rain_cum.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2, 
            file = paste0(out_dir, "01_F2_rain_cum.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3, 
            file = paste0(out_dir, "01_F3_rain_cum.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together, 
            file = paste0(out_dir, "01_F_rain_cum.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

#############################################################################
compute_median_diff <- function(dt, tgt_col, diff_from="1979-2016"){
  #
  # Clean unwanted data
  #
  if (diff_from == "1979-2016"){
    unwanted_period <- "1950-2005"
    } else {
    unwanted_period <- "1979-2016"
  }

  dt <- dt %>% 
        filter(time_period != unwanted_period & 
               time_period != "2006-2025") %>% 
        data.table()

  if ("evap" %in% colnames(dt)){
    dt <- within(dt, remove(evap, runoff, base_flow, run_p_base))
  }
  if ("wtr_yr" %in% colnames(dt)) {
    dt <- within(dt, remove(wtr_yr))
  }
  dt <- within(dt, remove(year, month, day, precip, cluster))

  ################################################
  #
  # We need to do the following lines in order
  # to make the subtraction within groups work.
  #
  dt_hist <- dt %>% 
             filter(time_period == diff_from) %>%
             select(-c("emission")) %>%
             unique()%>%
             data.table()

  dt_hist$emission <- "hist"

  dt <- dt %>% filter(time_period != diff_from) %>% data.table()
  dt <- rbind(dt, dt_hist)
  
  # med_per_loc_mod_TP_em <- data.frame(dt) %>% 
  #                          group_by(location, time_period, 
  #                                    emission, model) %>% 
  #                          summarise(medians = median(get(tgt_col)))  %>% 
  #                          data.table()
  # print(colnames(dt))
  med_per_loc_mod_TP_em <- dt[, .( tgt_col = median(get(tgt_col))), 
                               by = c("location", "time_period", 
                                      "emission", "model")]
  setnames(med_per_loc_mod_TP_em, old="tgt_col", new="medians")

  median_diffs <- med_per_loc_mod_TP_em %>%
                  group_by(location) %>%
                  mutate(diff = medians - medians[time_period == diff_from])%>%
                  data.table()

  median_diffs <- median_diffs %>% 
                  filter(time_period != diff_from)%>%
                  data.table()

  # to do percentages
  # hist_meds_tl <- med_per_loc_mod_TP_em %>% filter(time_period == diff_from) %>% data.table()
  # hist_meds_tl <- within(hist_meds_tl, remove(time_period, emission, model))
  # setnames(hist_meds_tl, old=c("medians"), new="hist_median")
  # median_diffs <- merge(median_diffs, hist_meds_tl, by="location", all.x=T)
  median_diffs$hist_median <- median_diffs$medians +  median_diffs$diff
  
  median_diffs$perc_diff <- (median_diffs$diff * 100) / (median_diffs$hist_median)
  return (list(med_per_loc_mod_TP_em, median_diffs))
}

modeled_F1 <- rbind(modeled_F1, obs)
modeled_F2 <- rbind(modeled_F2, obs)
modeled_F3 <- rbind(modeled_F3, obs)
all_together <- rbind(all_together, obs)

modeled_F1_rain_meds <- compute_median_diff(dt=modeled_F1, tgt_col="annual_cum_rain", diff_from="1979-2016")
modeled_F2_rain_meds <- compute_median_diff(dt=modeled_F2, tgt_col="annual_cum_rain", diff_from="1979-2016")
modeled_F3_rain_meds <- compute_median_diff(dt=modeled_F3, tgt_col="annual_cum_rain", diff_from="1979-2016")
all_together_rain_meds <- compute_median_diff(dt=all_together, tgt_col="annual_cum_rain", diff_from="1979-2016")

modeled_F1_rain_diffs <- modeled_F1_rain_meds[[2]]
modeled_F1_rain_meds <- modeled_F1_rain_meds[[1]]

modeled_F2_rain_diffs <- modeled_F2_rain_meds[[2]]
modeled_F2_rain_meds <- modeled_F2_rain_meds[[1]]

modeled_F3_rain_diffs <- modeled_F3_rain_meds[[2]]
modeled_F3_rain_meds <- modeled_F3_rain_meds[[1]]

all_together_rain_diffs <- all_together_rain_meds[[2]]
all_together_rain_meds <- all_together_rain_meds[[1]]
###########################################################################
#
# Test if having three future time periods will change anything
all_together_1 <- all_together_rain_meds %>% filter(time_period == "2026-2050") %>% data.table()
all_together_2 <- all_together_rain_meds %>% filter(time_period == "2051-2075") %>% data.table()
all_together_3 <- all_together_rain_meds %>% filter(time_period == "2076-2099") %>% data.table()

modeled_F1_rain_meds1 <- modeled_F1_rain_meds %>% filter(time_period == "2026-2050") %>% data.table()
modeled_F2_rain_meds2 <- modeled_F2_rain_meds %>% filter(time_period == "2051-2075") %>% data.table()
modeled_F3_rain_meds3 <- modeled_F3_rain_meds %>% filter(time_period == "2076-2099") %>% data.table()

col_order_1 <- colnames(modeled_F1_rain_meds)
setcolorder(modeled_F1_rain_meds1, col_order_1); 
setcolorder(modeled_F2_rain_meds2, col_order_1); 
setcolorder(modeled_F3_rain_meds3, col_order_1); 
setcolorder(all_together_1, col_order_1); 
setcolorder(all_together_2, col_order_1); 
setcolorder(all_together_3, col_order_1); 

all.equal(all_together_1, modeled_F1_rain_meds1)
all.equal(all_together_2, modeled_F2_rain_meds2)
all.equal(all_together_3, modeled_F3_rain_meds3)
########################################################################

write.table(modeled_F1_rain_meds, 
            file = paste0(out_dir, "02_F1_rain_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2_rain_meds, 
            file = paste0(out_dir, "02_F2_rain_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3_rain_meds, 
            file = paste0(out_dir, "02_F3_rain_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together_rain_meds, 
            file = paste0(out_dir, "02_F_rain_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F1_rain_diffs, 
            file = paste0(out_dir, "03_F1_rain_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2_rain_diffs, 
            file = paste0(out_dir, "03_F2_rain_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3_rain_diffs, 
            file = paste0(out_dir, "03_F3_rain_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together_rain_diffs, 
            file = paste0(out_dir, "03_F_rain_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

modeled_F1_snow_meds <- compute_median_diff(dt=modeled_F1, tgt_col="annual_cum_snow", diff_from="1979-2016")
modeled_F2_snow_meds <- compute_median_diff(dt=modeled_F2, tgt_col="annual_cum_snow", diff_from="1979-2016")
modeled_F3_snow_meds <- compute_median_diff(dt=modeled_F3, tgt_col="annual_cum_snow", diff_from="1979-2016")
all_together_snow_meds <- compute_median_diff(dt=all_together, tgt_col="annual_cum_snow", diff_from="1979-2016")

modeled_F1_snow_diffs <- modeled_F1_snow_meds[[2]]
modeled_F1_snow_meds <- modeled_F1_snow_meds[[1]]

modeled_F2_snow_diffs <- modeled_F2_snow_meds[[2]]
modeled_F2_snow_meds <- modeled_F2_snow_meds[[1]]

modeled_F3_snow_diffs <- modeled_F3_snow_meds[[2]]
modeled_F3_snow_meds <- modeled_F3_snow_meds[[1]]

all_together_snow_diffs <- all_together_snow_meds[[2]]
all_together_snow_meds <- all_together_snow_meds[[1]]

write.table(modeled_F1_snow_meds, 
            file = paste0(out_dir, "04_F1_snow_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2_snow_meds, 
            file = paste0(out_dir, "04_F2_snow_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3_snow_meds, 
            file = paste0(out_dir, "04_F3_snow_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together_snow_meds, 
            file = paste0(out_dir, "04_F_snow_medians.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")


write.table(modeled_F1_snow_diffs, 
            file = paste0(out_dir, "05_F1_snow_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F2_snow_diffs, 
            file = paste0(out_dir, "05_F2_snow_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(modeled_F3_snow_diffs, 
            file = paste0(out_dir, "05_F3_snow_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

write.table(all_together_snow_diffs, 
            file = paste0(out_dir, "05_F_snow_med_diffs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")
#############################################################################


