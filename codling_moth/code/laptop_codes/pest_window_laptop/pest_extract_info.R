

data_path = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/for_bar_plots/"

######
###### Larva 50%
######

median_df <- data.frame(matrix(ncol = 6, nrow = 8))
col_names = c("ClimateGroup", "CountyGroup", 
              paste0("gen_1"),
              paste0("gen_2"),
              paste0("gen_3"),
              paste0("gen_4"))
colnames(median_df) <- col_names

median_df[1, 1] = "Warmer Areas"
median_df[2, 1] = "Warmer Areas"
median_df[3, 1] = "Warmer Areas"
median_df[4, 1] = "Warmer Areas"

median_df[5, 1] = "Cooler Areas"
median_df[6, 1] = "Cooler Areas"
median_df[7, 1] = "Cooler Areas"
median_df[8, 1] = "Cooler Areas"

median_df[1, 2] = "Historical"
median_df[2, 2] = "2040's"
median_df[3, 2] = "2060's"
median_df[4, 2] = "2080's"

median_df[5, 2] = "Historical"
median_df[6, 2] = "2040's"
median_df[7, 2] = "2060's"
median_df[8, 2] = "2080's"

###### rcp 45
file_name = "df_larva_50_45.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .65 & Gen_1$PercLarvaGen1 > .45] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .65 & Gen_2$PercLarvaGen2 > .45] 
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .65 & Gen_3$PercLarvaGen3 > .45] 
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .65 & Gen_4$PercLarvaGen4 > .45] 
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp45_50.csv"), row.names = F)
###############
############### rcp 85
###############
file_name = "df_larva_50_85.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .65 & Gen_1$PercLarvaGen1 > .45] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .65 & Gen_2$PercLarvaGen2 > .45] 
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .65 & Gen_3$PercLarvaGen3 > .45] 
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .65 & Gen_4$PercLarvaGen4 > .45] 
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp85_50.csv"), row.names = F)
####################################################################################
#######################
#######################   Larva 25%
#######################
####################################################################################
######
###### rcp 45
###### 
file_name = "df_larva_25_45.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .5] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
#############
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .5 ]
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
#############
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .5 ]
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .5 ]
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp45_25.csv"), row.names = F)
################################################################################
########################################
########################################   rcp 85
########################################
################################################################################
file_name = "df_larva_25_85.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .5] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
#############
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .5 ]
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
#############
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .5 ]
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .5 ]
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp85_25.csv"), row.names = F)
######################################################
################################# 
################################# Larva 75%
################################# 
######################################################

###### rcp 45
file_name = "df_larva_75_45.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .5] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
#############
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .5 ]
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
#############
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .5 ]
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .5 ]
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp45_75.csv"), row.names = F)


###### rcp 85

file_name = "df_larva_75_85.rds"
curr_data = readRDS(paste0(data_path, file_name))
curr_data$CountyGroup = as.character(curr_data$CountyGroup)
curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas Areas'
curr_median_df <- median_df
#############
############# GEN 1
Gen_1 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))
Gen_1 <- Gen_1[Gen_1$PercLarvaGen1 < .5] 
d <- Gen_1
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 3] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 3] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 3] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 3] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 2
#############
Gen_2 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen3", "PercLarvaGen4"))
Gen_2 <- Gen_2[Gen_2$PercLarvaGen2 < .5 ]
d <- Gen_2
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 4] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 4] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 4] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 4] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 3
#############
Gen_3 <- within(curr_data, remove("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen4"))
Gen_3 <- Gen_3[Gen_3$PercLarvaGen3 < .5 ]
d <- Gen_3
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 5] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 5] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 5] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 5] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear
#############
############# GEN 4
Gen_4 <- within(curr_data, remove("PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen1"))
Gen_4 <- Gen_4[Gen_4$PercLarvaGen4 < .5 ]
d <- Gen_4
medians <- aggregate(d[, "dayofyear"], list(d$ClimateGroup, d$CountyGroup), median)

curr_median_df[1, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[2, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[3, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear
curr_median_df[4, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Warmer Areas Areas", ]$dayofyear

curr_median_df[5, 6] <- medians[medians$Group.1 == "Historical" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[6, 6] <- medians[medians$Group.1 == "2040's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[7, 6] <- medians[medians$Group.1 == "2060's" & medians$Group.2=="Cooler Areas", ]$dayofyear
curr_median_df[8, 6] <- medians[medians$Group.1 == "2080's" & medians$Group.2=="Cooler Areas", ]$dayofyear

write.csv(curr_median_df, paste0(data_path, "rcp85_75.csv"), row.names = F)



