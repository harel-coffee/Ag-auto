library(data.table)
library(dplyr)
source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)

 
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                   "06_list_of_double_cropped_plants/")

param_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/")

potential_plants <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

################################################################################################################
double_SEOS3 <- read.csv(paste0(data_dir, "extended_double_cropped_plants_SEOS3.csv"))
extended_double_cropped_plants_SEOS3 <- double_SEOS3

double_SEOS3 <- filter_out_non_irrigated_datatable(double_SEOS3)


double_SEOS3 <- double_SEOS3 %>%
                group_by(county, image_year, SG_params, CropTyp) %>%
                summarize(acr_sum = sum(Acres, na.rm = TRUE)) %>%
                data.table()

double_SEOS3 <- double_SEOS3 %>%
                filter(CropTyp %in% potential_plants$Crop_Type) %>%
                data.table()

double_SEOS3_73 <- double_SEOS3 %>%
                   filter(SG_params == 73) %>%
                   data.table()

double_SEOS3_73 <- setorderv(double_SEOS3_73, c("image_year", "acr_sum"), c(1, -1))

write.csv(double_SEOS3_73, 
          file = paste0(data_dir, "double_SEOS3_73_summary.csv"))


############
############ Top 5 counties by Acreage
############

double_SEOS3_73_2016 <- double_SEOS3_73 %>% 
                        filter(image_year == 2016) %>%
                        data.table()

double_SEOS3_73_2017 <- double_SEOS3_73 %>% 
                        filter(image_year == 2017) %>%
                        data.table()

double_SEOS3_73_2018 <- double_SEOS3_73 %>% 
                        filter(image_year == 2018) %>%
                        data.table()

double_SEOS3_73_2016_top5 <- double_SEOS3_73_2016 %>% 
                             group_by(county) %>%
                             summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                             data.table()
double_SEOS3_73_2016_top5 <- setorderv(double_SEOS3_73_2016_top5, c("acr_sum"), c(-1))
head(double_SEOS3_73_2016_top5 , 5)

double_SEOS3_73_2017_top5 <- double_SEOS3_73_2017 %>% 
                             group_by(county) %>%
                             summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                             data.table()
double_SEOS3_73_2017_top5 <- setorderv(double_SEOS3_73_2017_top5, c("acr_sum"), c(-1))
head(double_SEOS3_73_2017_top5 , 5)

double_SEOS3_73_2018_top5 <- double_SEOS3_73_2018 %>% 
                             group_by(county) %>%
                             summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                             data.table()
double_SEOS3_73_2018_top5 <- setorderv(double_SEOS3_73_2018_top5, c("acr_sum"), c(-1))
head(double_SEOS3_73_2018_top5 , 5)



################################################################################################################
#######
####### 2016
#######
MAtt_SOS03_2016_73 <- MAtt_SOS03 %>% 
                      filter(SF_year == 2016 & SG_params == 73) %>% 
                      data.table()

MAtt_SOS03_2016_73 <- setorderv(MAtt_SOS03_2016_73, c("acr_sum"), c(-1))

write.csv(MAtt_SOS03_2016_73, 
          file = paste0(data_dir, "MAtt_SOS03_2016_73.csv"))

#######
####### 2017
#######
MAtt_SOS03_2017_73 <- MAtt_SOS03 %>% 
                      filter(SF_year == 2017 & SG_params == 73) %>% 
                      data.table()

MAtt_SOS03_2017_73 <- setorderv(MAtt_SOS03_2017_73, c("acr_sum"), c(-1))

write.csv(MAtt_SOS03_2017_73, 
          file = paste0(data_dir, "MAtt_SOS03_2017_73.csv"))




#######
####### 2018
#######
MAtt_SOS03_2018_73 <- MAtt_SOS03 %>% 
                      filter(SF_year == 2018 & SG_params == 73) %>% 
                      data.table()

MAtt_SOS03_2018_73 <- setorderv(MAtt_SOS03_2018_73, c("acr_sum"), c(-1))

write.csv(MAtt_SOS03_2018_73, 
          file = paste0(data_dir, "MAtt_SOS03_2018_73.csv"))



############
############ Top 5 counties by Acreage
############

MAtt_SOS03_2016_73_top5 <- MAtt_SOS03_2016_73 %>% 
                           group_by(county) %>%
                           summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                           data.table()
MAtt_SOS03_2016_73_top5 <- setorderv(MAtt_SOS03_2016_73_top5, c("acr_sum"), c(-1))
head(MAtt_SOS03_2016_73_top5 , 5)




MAtt_SOS03_2017_73_top5 <- MAtt_SOS03_2017_73 %>% 
                           group_by(county) %>%
                           summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                           data.table()
MAtt_SOS03_2017_73_top5 <- setorderv(MAtt_SOS03_2017_73_top5, c("acr_sum"), c(-1))
head(MAtt_SOS03_2017_73_top5 , 5)


MAtt_SOS03_2018_73_top5 <- MAtt_SOS03_2018_73 %>% 
                           group_by(county) %>%
                           summarize(acr_sum = sum(acr_sum, na.rm = TRUE)) %>%
                           data.table()
MAtt_SOS03_2018_73_top5 <- setorderv(MAtt_SOS03_2018_73_top5, c("acr_sum"), c(-1))
head(MAtt_SOS03_2018_73_top5 , 5)