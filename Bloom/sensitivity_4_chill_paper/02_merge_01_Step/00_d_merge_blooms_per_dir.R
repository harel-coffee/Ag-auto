.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
###############################################################
print(paste0("getwd(): ", getwd()))

main_out <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/01_modeled_bloom/"
current_dir <- gsub(x = getwd(),
                    pattern = "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/01_modeled_bloom/",
                    replacement = "")
out_dir <- paste0(main_out, current_dir, "/")
print(paste0("current_dir: ", current_dir))

list_of_files <- dir()
# remove filenames that aren't data
list_of_files <- list_of_files[grep(pattern = "bloom_",
                        x = list_of_files)]

all_dt <- data.table()
print("line 24")
print(length(list_of_files))
print (list_of_files[1])
for (file in list_of_files){
  curr_dt <- data.table(readRDS(file))
  all_dt <- rbind(all_dt, curr_dt)

}

all_dt$time_period <- "future"
print (dim(all_dt))
saveRDS(all_dt, paste0(out_dir, "/all_dt.rds"))

end_time <- Sys.time()
print( end_time - start_time)



