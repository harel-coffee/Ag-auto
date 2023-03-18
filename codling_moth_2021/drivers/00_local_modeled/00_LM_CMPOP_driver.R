#######!/share/apps/R-3.2.2_gcc/bin/Rscript

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(lubridate)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)
options(digits=9)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

raw_data_dir = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/"
write_path =   "/data/hydro/users/Hossein/codling_moth_2021/00_LM_CMPOP/"
param_dir  =   "/home/hnoorazar/codling_moth_2021/parameters/"

models = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")

file_prefix = "data_"
locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations = locations$location

time_period = list("1979-2015", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

for (emission in c("rcp85", "rcp45")){
  for (model in models){
    for(location in locations) {  
      filename = paste0(model, "/", emission, "/", file_prefix, location)
      start_year = 2006
      end_year = 2099

      temp <- produce_CMPOP_local(filename = filename, 
                                  noVariables = 4, # modeled data have 4 columns
                                  input_folder = raw_data_dir, 
                                  param_dir=param_dir,
                                  cod_moth_param_name = "CodlingMothparameters.txt",
                                  scale_shift = 0,
                                  start_year = start_year, end_year = end_year,
                                  lower=10, upper=31.11,
                                  location = location, category = model)    

      write_dir = paste0(write_path, model, "/", emission)
      dir.create(file.path(write_dir), recursive = TRUE)
      write.table(temp, 
                  file = paste0(write_dir, "/CMPOP_", location), 
                  sep = ",", 
                  row.names = FALSE, 
                  col.names = TRUE)
    }
  }
}

