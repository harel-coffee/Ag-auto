# #!/share/apps/R-3.2.2_gcc/bin/Rscript  # old from 2018

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

param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

dir_base = "/data/hydro/users/Hossein/codling_moth_2021/"
input_dir = paste0(dir_base, "01_combined_files/")
write_dir = paste0(dir_base, "02_Diap_Gens_Vertdd/")
dir.create(file.path(write_dir), recursive = TRUE)

# args = commandArgs(trailingOnly=TRUE)
# emission = args[1]
# file_name = paste0("combined_CMPOP_", emission)

file_names = c("LM_combined_CMPOP_rcp45", 
               "LM_combined_CMPOP_rcp85",
               "LO_combined_CMPOP_observed")

all_rel_data = data.table()
all_abs_data = data.table()
all_plot_data = data.table()

for (file_name in file_names){
    
    output <- diapause_abs_rel(input_dir, paste0(file_name, ".rds"), param_dir)

    RelData <- data.table(output[[1]])
    AbsData <- data.table(output[[2]])
    sub1 = data.table(output[[3]])
    # pre_diap_plot = data.table(output[[4]])

    # saveRDS(RelData, paste0(write_dir, "diapause_rel_", file_name))
    # saveRDS(AbsData, paste0(write_dir, "diapause_abs_", file_name))
    # saveRDS(sub1,    paste0(write_dir, "diapause_plot_", file_name))

    emission <- tail(strsplit(file_name, "_")[[1]], 1)
    if (emission == "rcp45"){
        em = "RCP 4.5"
       } else if (emission == "rcp85"){
        em = "RCP 8.5"
       } else if (emission == "observed"){
        em = "Observed"
    }

    RelData$emission <- em
    AbsData$emission <- em
    sub1$emission <- em

    all_rel_data <- rbind(all_rel_data, RelData)
    all_abs_data <- rbind(all_abs_data, AbsData)
    all_plot_data <- rbind(all_plot_data, sub1)
}

# saveRDS(pre_diap_plot, paste0(write_dir, "pre_diap_plot_", emission, ".rds"))
L = c("1979-2015", "2040s", "2060s", "2080s")
all_rel_data$time_period <- factor(all_rel_data$time_period, levels = L, ordered = TRUE)
all_abs_data$time_period <- factor(all_abs_data$time_period, levels = L, ordered = TRUE)
all_plot_data$time_period <- factor(all_plot_data$time_period, levels = L, ordered = TRUE)

#
# had forgotten the convenction I had established for names: 
# diapause_rel_combined_CMPOP means diapause_rel came from combined_CMPOP
# This makes it easier to track stuff.
#
saveRDS(all_rel_data, paste0(write_dir, "diapause_rel_combined_CMPOP.rds"))
saveRDS(all_abs_data, paste0(write_dir, "diapause_abs_combined_CMPOP.rds"))
saveRDS(all_plot_data, paste0(write_dir, "diapause_plot_combined_CMPOP.rds"))


