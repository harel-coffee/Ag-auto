.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

make_unique(input_dir, param_dir, location_group_name = "/LocationGroups.csv")

