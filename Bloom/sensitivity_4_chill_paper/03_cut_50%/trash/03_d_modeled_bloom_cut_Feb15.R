.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
##################################################################
##                                                              ##
##              Terminal/shell/bash arguments                   ##
##                                                              ##
##################################################################
#
# Define main output path
#
param_dir <- "/home/hnoorazar/bloom_codes/parameters/"
in_dir <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/02_merged_01_Step/"
out_dir <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/cut_50/"

if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = file.path(out_dir), recursive=T)
}
##################################################################


cut_off=0.5


met_data <- data.table(readRDS(paste0(in_dir, "future_blooms.rds")))

met_data <- met_data %>% 
            filter(year >= 2026) %>% 
            data.table()

met_data <- subset(met_data, 
                   select = c("year", "month", "day",
                              "location",
                              "model", "emission",
                              "vert_Cum_dd", "start_accum_date", "dist_mean"))

met_data$dayofyear <- 1 # dummy
met_data[, dayofyear := cumsum(dayofyear), 
           by=list(year, location, model, emission, start_accum_date, dist_mean)]

# we have to do a correction, since DoY is computed 
# based on start date of heat accumulation
met_data$dayofyear <- met_data$start_accum_date + met_data$dayofyear - 1

met_data <- met_data %>% 
            filter(month==2 & day == 15) %>% 
            data.table()

unmeltead <- met_data
met_data <- melt(met_data, id.vars = c("location", 
                                        "model", "emission",
                                        "year", "month", "day", 
                                        "dayofyear", "start_accum_date", "dist_mean"),
                 variable.name = "col_type")

setnames(met_data, old=c("value"), new=c("vert_Cum_dd"))


limited_locations <- read.csv(paste0(param_dir, "limited_locations.csv"))
limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

met_data <- dplyr::left_join(x = met_data, y = limited_locations, by = "location")

met_data_F1 <- met_data %>% 
               filter(year >= 2026 & year <= 2050) %>% 
               data.table()

met_data_F2 <- met_data %>% 
               filter(year >= 2051 & year <= 2075) %>% 
               data.table()

met_data_F3 <- met_data %>% 
               filter(year >= 2076) %>% 
               data.table()

met_data_F1$time_period <- "2026-2050"
met_data_F2$time_period <- "2051-2075"
met_data_F3$time_period <- "2076-2099"

met_data <- rbind(met_data_F1, met_data_F2, met_data_F3)

saveRDS(object = met_data,
        file = paste0(out_dir, "/heat_Feb15.rds"))


unmeltead <- dplyr::left_join(x = unmeltead, y = limited_locations, by = "location")

unmeltead_F1 <- unmeltead %>% 
               filter(year >= 2026 & year <= 2050) %>% 
               data.table()

unmeltead_F2 <- unmeltead %>% 
               filter(year >= 2051 & year <= 2075) %>% 
               data.table()

unmeltead_F3 <- unmeltead %>% 
                filter(year >= 2076) %>% 
                data.table()

unmeltead_F1$time_period <- "2026-2050"
unmeltead_F2$time_period <- "2051-2075"
unmeltead_F3$time_period <- "2076-2099"

unmeltead <- rbind(unmeltead_F1, unmeltead_F2, unmeltead_F3)

saveRDS(object = unmeltead,
        file = paste0(out_dir, "/heat_Feb15_unmelt.rds"))

end_time <- Sys.time()
print( end_time - start_time)

