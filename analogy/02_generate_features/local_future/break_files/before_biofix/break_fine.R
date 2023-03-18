.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

#########################################################################
main_in <- "/data/hydro/users/Hossein/analog/local/ready_features/one_file_4_all_locations/"
main_out <- "/data/hydro/users/Hossein/analog/local/ready_features/broken_down_location_year_level/"

dirs <- c("averaged_data", "bcc_csm1_1_m", "BNU_ESM", "CanESM2", "CNRM_CM5", "GFDL_ESM2G", "GFDL_ESM2M")

file_list = list.files(path = main_in, pattern=".rds")

for (file in file_list){
  curr_file <- data.table(readRDS(paste0(main_in, file)))
  model = unlist(strsplit(file, "_"))[2]
  carbon_type <- unlist(strsplit(file, "_"))[3]
  carbon_type <- substr(x = carbon_type, start = 1, stop = 5)

  all_locations <- unique(curr_file$location)
  all_years <- unique(curr_file$year)
  counter = 1
  for (loc in all_locations){
    counter = counter + 1
    for (yr in all_years){
      curr_loc_year_data <- curr_file %>% filter(location == loc & year == yr)
      
      curr_out = file.path(main_out, model, carbon_type)
      if (dir.exists(file.path(curr_out)) == F){
        dir.create(path = curr_out, recursive = T)
      }
      if (counter == 1){
        print ("Line 48")
        print (curr_out)
        print (loc)
        print (yr)
        print ("________________________")
      }
      saveRDS(curr_loc_year_data, paste0(curr_out, "/feat_", loc, "_", yr, ".rds"))
    }
  }
}

