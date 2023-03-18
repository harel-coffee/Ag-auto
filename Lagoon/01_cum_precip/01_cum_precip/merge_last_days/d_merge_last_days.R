.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

###########################################################

main_in <- "/data/hydro/users/Hossein/lagoon/01_storm_cumPrecip/cum_precip/"
out_dir <- paste0(main_in, "last_days/")
param_dir <- "/home/hnoorazar/lagoon_codes/parameters/"
###########################################################
subdir <- c(# "annual/", 
            "chunky/"# , 
            # "monthly/", 
            # "wtr_yr/"
            )

for (sub in subdir){
  in_dir <- file.path(paste0(main_in, sub))
  files_list <- list.files(path=in_dir, pattern="LD")
  last_days <- data.table()
  if (sub == "annual/"){
     name_pref <- "ann_"
     } else if (sub == "chunky/"){
       name_pref <- "Sept_March_"
     } else if (sub == "monthly/"){
       name_pref <- "month_"
     } else if (sub == "wtr_yr/"){
       name_pref <- "wtr_yr_"
  }
  print (files_list)
  for (file in files_list){
    A <- readRDS(paste0(in_dir, file)) %>% data.table()
    print (paste0(in_dir, file))
    if ("observed.rds" %in%  unlist(strsplit(file, "_"))){
      print (file)
      A_45 <- A
      A_85 <- A
      A_45$emission <- "RCP 4.5"
      A_85$emission <- "RCP 8.5"
      A <- rbind(A_45, A_85)
      print (unique(A$emission))
      rm(A_45, A_85)
    }
    if ("hist.rds" %in%  unlist(strsplit(file, "_"))){
      print (file)
      A_45 <- A
      A_85 <- A
      A_45$emission <- "RCP 4.5"
      A_85$emission <- "RCP 8.5"
      A <- rbind(A_45, A_85)
      print (unique(A$emission))
      rm(A_45, A_85)
    }
    last_days <- rbind(last_days, A)
    rm(A)
  }
  saveRDS(last_days, paste0(in_dir, "/", name_pref, "all_last_days.rds"))
  rm(last_days)
}

end_time <- Sys.time()
print( end_time - start_time)


