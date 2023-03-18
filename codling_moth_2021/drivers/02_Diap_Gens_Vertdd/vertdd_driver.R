# #!/share/apps/R-3.2.2_gcc/bin/Rscript  # old from 2018

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
options(digits=9)



source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

dir_base = "/data/hydro/users/Hossein/codling_moth_2021/"
input_dir = paste0(dir_base, "01_combined_files/")
write_dir = paste0(dir_base, "02_Diap_Gens_Vertdd/")


low_temp = 4.5
up_temp = 24.28
# lower = 10 # 50 F
# upper = 31.11 # 88 F


# args = commandArgs(trailingOnly=TRUE)
# file_name = args[1]

file_names = c("LM_combined_CMPOP_rcp45", 
               "LM_combined_CMPOP_rcp85",
               "LO_combined_CMPOP_observed")

all_vertdd = data.table()

for (file_name in file_names){
    vertdd = generate_vertdd(input_dir, paste0(file_name, ".rds"), lower_temp=low_temp, upper_temp=up_temp)
    # output_name = paste0("vertdd_", file_name)
    # saveRDS(vertdd, paste0(write_dir, output_name, ".rds"))

    emission <- tail(strsplit(file_name, "_")[[1]], 1)
    if (emission == "rcp45"){
        em = "RCP 4.5"
       } else if (emission == "rcp85"){
        em = "RCP 8.5"
       } else if (emission == "observed"){
        em = "Observed"
    }
    vertdd$emission <- em
    all_vertdd = rbind(all_vertdd, vertdd)

}

L = c("1979-2015", "2040s", "2060s", "2080s")
all_vertdd$time_period <- factor(all_vertdd$time_period, levels = L, ordered = TRUE)
#
# had forgotten the convenction I had established for names: vertdd_combined_CMPOP means vertdd came from combined_CMPOP
# This makes it easier to track stuff.
#
saveRDS(all_vertdd, paste0(write_dir, "vertdd_combined_CMPOP.rds"))



