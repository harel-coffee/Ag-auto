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
dir.create(file.path(write_dir), recursive = TRUE)

# args = commandArgs(trailingOnly=TRUE)
# emission = args[1]
# file_name = paste0("combined_CMPOP_", emission, ".rds")

file_names = c("LM_combined_CMPOP_rcp45", 
               "LM_combined_CMPOP_rcp85",
               "LO_combined_CMPOP_observed")

gen_by_aug <- data.table()
gen_by_nov <- data.table()

for (file_name in file_names){
    output = generations_func(input_dir, paste0(file_name, ".rds"))
    generations_Aug = data.table(output[[1]])
    generations_Nov = data.table(output[[2]])

    emission <- tail(strsplit(file_name, "_")[[1]], 1)
    if (emission == "rcp45"){
        em = "RCP 4.5"
       } else if (emission == "rcp85"){
        em = "RCP 8.5"
       } else if (emission == "observed"){
        em = "Observed"
    }
    generations_Aug$emission <- em
    generations_Nov$emission <- em

    gen_by_aug <- rbind(gen_by_aug, generations_Aug)
    gen_by_nov <- rbind(gen_by_nov, generations_Nov)

    # saveRDS(generations_Aug, paste0(write_dir, "generations_Aug_", file_name))
    # saveRDS(generations_Nov, paste0(write_dir, "generations_Nov_", file_name))
}

L = c("1979-2015", "2040s", "2060s", "2080s")
gen_by_aug$time_period <- factor(gen_by_aug$time_period, levels = L, ordered = TRUE)
gen_by_nov$time_period <- factor(gen_by_nov$time_period, levels = L, ordered = TRUE)

#
# had forgotten the convenction I had established for names: 
# generations_Aug_combined_CMPOP.rds means generations_Aug came from combined_CMPOP.
# This makes it easier to track stuff.
#

saveRDS(gen_by_aug, paste0(write_dir, "generations_Aug_combined_CMPOP.rds"))
saveRDS(gen_by_nov, paste0(write_dir, "generations_Nov_combined_CMPOP.rds"))

