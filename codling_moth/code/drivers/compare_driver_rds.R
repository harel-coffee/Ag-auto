#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
#library(reshape2)
library(dplyr)
library(foreach)
library(doParallel)
library (bigmemory)
library (biganalytics)
#library(iterators)
source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)


input_1_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
input_1_name = "vertdd_combined_CMPOP_rcp45.rds"

input_2_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
input_2_name = "allData_vertdd_new_rcp45.rds"

write_dir = input_1_dir

my_file_name = paste0(input_1_dir, input_1_name)
girid_file_name = paste0(input_2_dir, input_2_name)

my_file = readRDS(my_file_name)
girid_file = readRDS(girid_file_name)

my_file_first = my_file[1:20399545]
my_file_second = my_file[20399546:40799091]
my_file_third = my_file[40799091:61198635]
rm(my_file)


girid_file_first = girid_file[1:20399545]
girid_file_second = girid_file[20399546:40799091]
girid_file_third = girid_file[40799091:61198635]
rm(girid_file)

compare_result1 = grep(x = my_file_first == girid_file_first, pattern = "TRUE")
compare_result2 = grep(x = my_file_second == girid_file_second, pattern = "TRUE")
compare_result3 = grep(x = my_file_third == girid_file_third, pattern = "TRUE")


write.table(data.frame(compare_result, my_file_name, girid_file_name, 
	                  compare_result1, compare_result2, compare_result3), 
                      paste0(write_dir, "compare_result_rds" , ".txt"), 
            sep="\t")

