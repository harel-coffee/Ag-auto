.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
# 
#                   Directories
# 
################################################################################
main_dir <- "/data/hydro/users/Hossein/analog/"

main_us_dir <- file.path(main_dir, "usa/ready_features/")
main_local_dir <- file.path(main_dir, "local/ready_features/one_file_4_all_locations/")
main_out <- file.path(main_dir, "03_analogs/no_gen_3/avg_vs_avg/")
################################################################################
################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt_usa <- data.table(readRDS(paste0(main_us_dir, "all_data_usa.rds")))
all_dt_usa <- within(all_dt_usa, remove(treatment))
ICV <- all_dt_usa

# take averages
all_dt_usa <- all_dt_usa %>%
              group_by(location) %>%
              summarise_at(.funs = funs(mean(., na.rm=TRUE)), vars(medianDoY:mean_gdd))%>% 
              data.table()

# In the line above year and ClimateScenario are gone.
# We add it back to the data frame, because the function in
# the core needs them to be there, so data have the same dimension.
all_dt_usa$year = "avg_hist"
all_dt_usa$ClimateScenario = "observed"

########################################################
# avg data we read below, are averages over the
# models. In this driver we want to find nearest neighbors
# for averages over models, and also averages over years
# for three time periods 2026-2050, 2051-2075 and 2076-2095
#
########################################################
carbon_types =  c("rcp45", "rcp85") #
no_nghbrs = 500
precip = TRUE

for (emission_type in carbon_types){
  ###########################################################################
  # create subdirectory for specific emission types
  out_dir = file.path(main_out, "precip", no_nghbrs, emission_type)
  if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T)}

  file_name <- paste0("data_avgeraged_", emission_type, ".rds")
  local_dt <- data.table(readRDS(paste0(main_local_dir, file_name)))
  
  local_dt_26_50 <- local_dt %>% filter(year<=2050)
  local_dt_51_75 <- local_dt %>% filter(year>=2051 & year<=2075)
  local_dt_76_95 <- local_dt %>% filter(year>=2076)

  local_dt_26_50 <- local_dt_26_50 %>%
                    group_by(location) %>%
                    summarise_at(.funs = funs(mean(., na.rm=TRUE)), vars(medianDoY:mean_gdd))%>% 
                    data.table()
  local_dt_26_50$year = "2026_2050"
  local_dt_26_50$ClimateScenario <- "ensembe_mean"

  local_dt_51_75 <- local_dt_51_75 %>%
                    group_by(location) %>%
                    summarise_at(.funs = funs(mean(., na.rm=TRUE)), vars(medianDoY:mean_gdd))%>% 
                    data.table()
  local_dt_51_75$year = "2051_2075"
  local_dt_51_75$ClimateScenario <- "ensembe_mean"

  local_dt_76_95 <- local_dt_76_95 %>%
                    group_by(location) %>%
                    summarise_at(.funs = funs(mean(., na.rm=TRUE)), vars(medianDoY:mean_gdd))%>% 
                    data.table()
  local_dt_76_95$year = "2076_2095"
  local_dt_76_95$ClimateScenario <- "ensembe_mean"

  ################################################################################
  # 
  #                   Set parameters and run the code
  # 
  ################################################################################
  information_26_50 = find_NN_info_W4G_ICV(ICV=ICV, historical_dt=all_dt_usa, future_dt=local_dt_26_50, n_neighbors=no_nghbrs, precipitation=precip)
  information_51_75 = find_NN_info_W4G_ICV(ICV=ICV, historical_dt=all_dt_usa, future_dt=local_dt_51_75, n_neighbors=no_nghbrs, precipitation=precip)
  information_76_95 = find_NN_info_W4G_ICV(ICV=ICV, historical_dt=all_dt_usa, future_dt=local_dt_76_95, n_neighbors=no_nghbrs, precipitation=precip)

  ################################################################################
  ############
  ############ information_26_50
  ############
  NN_dist_tb = information_26_50[[1]]
  NN_loc_year_tb = information_26_50[[2]]
  NN_sigma_tb = information_26_50[[3]]

  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_avg_26_50.rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_avg_26_50.rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_avg_26_50.rds"))
  ################################################################################
  ############
  ############ information_51_75
  ############
  NN_dist_tb = information_51_75[[1]]
  NN_loc_year_tb = information_51_75[[2]]
  NN_sigma_tb = information_51_75[[3]]

  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_avg_51_75.rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_avg_51_75.rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_avg_51_75.rds"))
  ################################################################################
  ############
  ############ information_76_95
  ############
  NN_dist_tb = information_76_95[[1]]
  NN_loc_year_tb = information_76_95[[2]]
  NN_sigma_tb = information_76_95[[3]]

  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_avg_76_95.rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_avg_76_95.rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_avg_76_95.rds"))
}




