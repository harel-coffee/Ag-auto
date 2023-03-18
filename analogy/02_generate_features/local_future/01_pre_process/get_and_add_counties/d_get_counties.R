.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

# This fukcing function is created because unique, ~duplicate,
# nothing could work! So, we first separate the data
# then bind it together with another function.
# then compute diapause stuff
#

in_file = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/combined_CMPOP.rds"
out_dir = "/data/hydro/users/Hossein/analog/usa/data_bases/"

counties <- get_county(input_fill_add=in_file)

saveRDS(counties, paste0(out_dir, "counties.rds"))