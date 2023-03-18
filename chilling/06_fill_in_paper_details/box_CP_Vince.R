rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_core = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
source_path_plot = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_plot)
source(source_path_core)

##########################################################################################
###                                                                                    ###
###                                    Driver                                          ###
###                                                                                    ###
##########################################################################################
plot_dir_base <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/box_CP_Vince/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
limited_locations <- read.csv(paste0(param_dir, "limited_locations.csv"), header=T, as.is=T)
limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
main_in <- "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
files_name = c("sept_summary_comp.rds")

datas = data.table(readRDS(paste0(main_in, "sept_summary_comp.rds")))


datas$emission[datas$emission=="rcp85"] = "RCP 8.5"
datas$emission[datas$emission=="rcp45"] = "RCP 4.5"

datas$time_period[datas$time_period == "1979-2015"] <- "Historical"
##########################################################################################
datas <- datas %>% filter(time_period != "1950-2005") %>% data.table()
datas <- datas %>% filter(time_period != "2006-2025") %>% data.table()

data_f <- datas %>% filter(time_period != "Historical")

data_h_rcp85 <- datas %>% filter(time_period == "Historical")
data_h_rcp45 <- datas %>% filter(time_period == "Historical")

data_h_rcp85$emission = "RCP 8.5"
data_h_rcp45$emission = "RCP 4.5"


datas = rbind(data_f, data_h_rcp45, data_h_rcp85)
rm(data_h_rcp45, data_h_rcp85, data_f)
##########################################################################################

######
###### Keep Locations of interest
######
datas <- datas %>% 
         filter(location %in% limited_locations$location) %>% 
         data.table()
#
# add city to the data
#
limited_locations <- within(limited_locations, remove(lat, long))

datas <- dplyr::left_join(x = datas, y = limited_locations, by = "location")

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

datas <- datas %>% 
         filter(city %in% ict) %>% 
         data.table()

datas$city <- factor(datas$city, levels = ict, order=TRUE)


ch_start = "Sept. 1"
#
# plot size
#
box_width = 12
box_height = 5

source_path_plot = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_plot)

# Eugene_safe_chill_sept_Apr_RCP_85


for (em in c("RCP 4.5", "RCP 8.5")){
  curr_dt = datas %>% filter(emission == em) %>% data.table()

  ###
  ###  Jan 1
  ###
  cp_apr <- acc_CP(data = curr_dt, colname = "sum_J1")
  output_name = paste0("CP_sept_Dec31_", gsub('\\.',  "", gsub(" ", "_", em)), ".png")
  plot_dir <- paste0(plot_dir_base, "/Sept_1_Dec31/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  ggsave(output_name, cp_apr, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=600)
  
  ###
  ###  Feb 1
  ###
  cp_apr <- acc_CP(data = curr_dt, colname = "sum_F1")
  output_name = paste0("CP_sept_Jan31_", gsub('\\.',  "", gsub(" ", "_", em)), ".png")
  plot_dir <- paste0(plot_dir_base, "/Sept_1_Jan31/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  ggsave(output_name, cp_apr, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=600)

  ###
  ###  Mar 1
  ###
  cp_apr <- acc_CP(data = curr_dt, colname = "sum_M1")
  output_name = paste0("CP_sept_Feb31_", gsub('\\.',  "", gsub(" ", "_", em)), ".png")
  plot_dir <- paste0(plot_dir_base, "/Sept_1_Feb31/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  ggsave(output_name, cp_apr, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=600)

  ###
  ###  Apr 1
  ###
  cp_apr <- acc_CP(data = curr_dt, colname = "sum_A1")
  output_name = paste0("CP_sept_Mar31_", gsub('\\.',  "", gsub(" ", "_", em)), ".png")
  plot_dir <- paste0(plot_dir_base, "/Sept_1_March31/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  ggsave(output_name, cp_apr, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=600)

}




