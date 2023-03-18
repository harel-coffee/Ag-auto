.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
##################################################################
##                                                              ##
##              Terminal/shell/bash arguments                   ##
##                                                              ##
##################################################################
#
# Define main output path
#
FB_out = "/data/hydro/users/Hossein/bloom/02_bloomCut_first_frost/"
main_out <- file.path(FB_out, "/frost/modeled/")
# 2. Pre-processing prep -----------------------------------------
# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern="historical", x=getwd())==T, TRUE, FALSE)

# Get current folder
pp <- "/data/hydro/users/Hossein/bloom/01_binary_to_bloom/modeled/"
current_dir <- gsub(x = getwd(),
                    pattern = pp,
                    replacement = "")
if (dir.exists(file.path(main_out, current_dir)) == F) {
  dir.create(path = file.path(main_out, current_dir), recursive=T)
}
current_model <- gsub("-", "_", basename(dirname(current_dir)))
current_emission <- basename(current_dir)
print("does this look right?")
print(file.path(main_out, current_dir))
print(paste0("current_dir is ", current_dir))
print(paste0("current_model is ", current_model))
print(paste0("current_emission is ", current_emission))

# 2d. get files in current folder
the_dir <- dir()
the_dir <- the_dir[grep(pattern = ".rds", x = the_dir)]

# 3. Process the data ---------------------------------------

all_frosts <- data.table()
for(file in the_dir){
  met_data <- data.table(readRDS(file))
  met_data <- trim_chill_calendar(met_data)
  first_frost <- find_1st_frost(met_data)
  all_frosts <- rbind(all_frosts, first_frost)
}
saveRDS(object=all_frosts,
        file=file.path(main_out, current_dir, 
                       paste0("/frost_",
                              current_model, "_", 
                              current_emission,
                              ".rds")))
end_time <- Sys.time()
print( end_time - start_time)

