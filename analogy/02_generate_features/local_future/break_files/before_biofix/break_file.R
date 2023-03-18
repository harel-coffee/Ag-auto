.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

#########################################################################
main_in = "/data/hydro/users/Hossein/analog/local/ready_features/one_file_4_all_locations/"
main_out <- "/data/hydro/users/Hossein/analog/local/ready_features/broken_down_location_level_coarse/"

print ("line 15 main_in:")
print (main_in)
print ("________________________")

files_list <- dir(file.path(main_in))
rds_files <- files_list[grep(pattern = ".rds", x = files_list)]

for (file in rds_files){
  model = unlist(strsplit(file, "_"))[2]
  carbon_type <- unlist(strsplit(unlist(strsplit(file, "_"))[3], '[.]'))[1]
  print ("line 25 main_in:")
  print (paste(carbon_type, model, sep=","))
  print ("________________________")

  curr_file <- data.table(readRDS(paste0(main_in, file)))
  all_locations = unique(curr_file$location)
  
  for (loc in all_locations){
    curr_loc_data <- curr_file %>% filter(location == loc)

    locname = gsub(x=gsub(x=loc, pattern = "-", replacement = ""), pattern = '[.]', replacement = '_')
    curr_out = file.path(main_out, carbon_type, model)
    print ("line 37 main_in:")
    print (curr_out)
    print ("________________________")
    
    if (dir.exists(file.path(curr_out)) == F){
      dir.create(path = curr_out, recursive = T)
    }
    saveRDS(curr_loc_data, paste0(curr_out, "/", paste0("feat_", locname), ".rds"))
  }

}

