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

#####################################################################################
param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

data_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_01_WSS_CM_CMPOP_Merged/"
output_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_02_WSS_diapause_generations/"
#******************************************************************************************************
#
#  The function diapause_abs_rel(.) is written for one emission 
#  and does not include shift column as well. So, we have to have a for-loop that
#  does this and then put those columns back in.
#

shifts = as.character(seq(0, 20, 1)/100)

########################################################################
######
######  observed - Once done. commented out
######

all_rel_data <- data.table()
all_abs_data <- data.table()
all_plot_data <- data.table()

CMPOP <- data.table(readRDS(paste0(data_dir, "combined_LO_CMPOP_WSS.rds")))

for (shipht in shifts){
  for (em in c("observed")){
    curr_cmpop <- CMPOP %>% 
                  filter(emission == em & shift == shipht) %>% 
                  data.table()

    output = diapause_abs_rel(curr_cmpop)

    RelData = data.table(output[[1]])
    AbsData = data.table(output[[2]])
    sub1    = data.table(output[[3]])

    RelData$emission <- em
    AbsData$emission <- em
    sub1$emission    <- em

    RelData$shift <- shipht
    AbsData$shift <- shipht
    sub1$shift    <- shipht

    all_rel_data <- rbind(all_rel_data, RelData)
    all_abs_data <- rbind(all_abs_data, AbsData)
    all_plot_data<- rbind(all_plot_data,sub1)
  }
}

saveRDS(all_rel_data,  paste0(output_dir, "LO_diap_rel_data.rds"))
saveRDS(all_abs_data,  paste0(output_dir, "LO_diap_abs_data.rds"))
saveRDS(all_plot_data, paste0(output_dir, "LO_diap_plot_data.rds"))



########################################################################
######
######  modeled
######

all_rel_data <- data.table()
all_abs_data <- data.table()
all_plot_data <- data.table()

CMPOP <- data.table(readRDS(paste0(data_dir, "combined_LM_CMPOP_WSS.rds")))

for (shipht in shifts){
  for (em in c("RCP 4.5", "RCP 8.5")){
    curr_cmpop <- CMPOP %>% 
                  filter(emission == em & shift == shipht) %>% 
                  data.table()

    output = diapause_abs_rel(curr_cmpop)

    RelData = data.table(output[[1]])
    AbsData = data.table(output[[2]])
    sub1    = data.table(output[[3]])

    RelData$emission <- em
    AbsData$emission <- em
    sub1$emission    <- em

    RelData$shift <- shipht
    AbsData$shift <- shipht
    sub1$shift    <- shipht

    all_rel_data <- rbind(all_rel_data, RelData)
    all_abs_data <- rbind(all_abs_data, AbsData)
    all_plot_data<- rbind(all_plot_data,sub1)
  }
}

saveRDS(all_rel_data,  paste0(output_dir, "LM_diap_rel_data.rds"))
saveRDS(all_abs_data,  paste0(output_dir, "LM_diap_abs_data.rds"))
saveRDS(all_plot_data, paste0(output_dir, "LM_diap_plot_data.rds"))


