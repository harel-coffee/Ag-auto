
read_data_dir = "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
CAHNR_data_dir = "/Users/hn/Desktop/Kirti/check_point/CAHNR/2015/"
file_name_45 = "combined_CM_rcp45.rds"
file_name_85 = "combined_CM_rcp85.rds"

col_names = c("AdultApr1", "AdultAug1" , "AdultDec1" , "AdultFeb1", 
	          "AdultJul1", "AdultJune1", "AdultMar1" , "AdultMay1",
	          "AdultNov1", "AdultOct1" , "AdultSep1" , "AGen1_0.25",
	          "AGen1_0.5", "AGen1_0.75", "AGen2_0.25","AGen2_0.5",
	          "AGen2_0.75","AGen3_0.25", "AGen3_0.5" , "AGen3_0.75",
	          "AGen4_0.25","AGen4_0.5" , "AGen4_0.75","Diapause",  
	          "Emergence", "LarvaApr1" , "LarvaAug1" , "LarvaDec1",
	          "LarvaFeb1", "LarvaJul1" , "LarvaJune1","LarvaMar1", 
	          "LarvaMay1", "LarvaNov1" , "LarvaOct1" , "LarvaSep1",
	          "LarvaSep15","LGen1_0.25", "LGen1_0.5" , "LGen1_0.75", 
	          "LGen2_0.25","LGen2_0.5" , "LGen2_0.75", "LGen3_0.25",
	          "LGen3_0.5", "LGen3_0.75", "LGen4_0.25", "LGen4_0.5",
	          "LGen4_0.75","location"  , "timeFrame" , "year")

combined_CM_rcp45 = data.table(readRDS(paste0(read_data_dir, file_name_45)))
names(combined_CM_rcp45)[names(combined_CM_rcp45) == "ClimateGroup"] = "timeFrame"
combined_CM_rcp45$location = paste0(combined_CM_rcp45$latitude, "_", combined_CM_rcp45$longitude)
combined_CM_rcp45$timeFrame <- as.factor(combined_CM_rcp45$timeFrame)
combined_CM_rcp45$timeFrame <- factor(combined_CM_rcp45$timeFrame, levels = levels(combined_CM_rcp45$timeFrame)[c(4,1,2,3)])
combined_CM_rcp45 = subset(combined_CM_rcp45, select=col_names)
saveRDS(combined_CM_rcp45, paste0(CAHNR_data_dir, "combinedData_rcp45.rds"))
rm(combined_CM_rcp45, file_name_45)

combined_CM_rcp85 = data.table(readRDS(paste0(read_data_dir, file_name_85)))
names(combined_CM_rcp85)[names(combined_CM_rcp85) == "ClimateGroup"] = "timeFrame"
combined_CM_rcp85$location = paste0(combined_CM_rcp85$latitude, "_", combined_CM_rcp85$longitude)
combined_CM_rcp85$timeFrame <- as.factor(combined_CM_rcp85$timeFrame)
combined_CM_rcp85$timeFrame <- factor(combined_CM_rcp85$timeFrame, levels = levels(combined_CM_rcp85$timeFrame)[c(4,1,2,3)])
combined_CM_rcp85 = subset(combined_CM_rcp85, select=col_names)
saveRDS(combined_CM_rcp85, paste0(CAHNR_data_dir, "combinedData.rds"))
rm(combined_CM_rcp85, file_name_85)

