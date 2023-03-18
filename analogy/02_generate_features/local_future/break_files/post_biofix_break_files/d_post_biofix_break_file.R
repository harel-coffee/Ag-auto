.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

#########################################################################
main_in = "/data/hydro/users/Hossein/analog/02_features_post_biofix/"
main_out <- paste0(main_in, "future_broken/")

rcp_types <- c("rcp45", "rcp85")

for (rcp in rcp_types){
  data <- data.table(readRDS(paste0(main_in, "CDD_precip_", rcp, ".rds")))
  all_models = unique(data$model)
  
  for (model_n in all_models){
    curr_file <- data %>% filter(model == model_n)
    all_locations = unique(curr_file$location)
    for (loc in all_locations){
      curr_loc_data <- curr_file %>% filter(location == loc)
      locname = gsub(x=gsub(x=loc, pattern = "-", replacement = ""), pattern = '[.]', replacement = '_')
      curr_out = file.path(main_out, rcp, model_n)
      print ("________________________")
      print (paste0("The rcp is ", rcp))
      print (paste0("The model is ", model_n))
      print ("line 37 main_in:")
      print (curr_out)
      print ("________________________")
      
      if (dir.exists(file.path(curr_out)) == F){
        dir.create(path = curr_out, recursive = T)
      }
      saveRDS(curr_loc_data, paste0(curr_out, "/", paste0("feat_", locname), ".rds"))
    }
  }
}

