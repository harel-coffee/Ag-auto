

bloom_rcp85$emission <- "RCP 8.5"
bloom_rcp45$emission <- "RCP 4.5"

bloom_rcp85 <- merge(bloom_rcp85, limited_cities, all.x=TRUE)
bloom_rcp45 <- merge(bloom_rcp45, limited_cities, all.x=TRUE)

bloom_rcp85 <- within(bloom_rcp85, remove(location, latitude, longitude, value, month, day))
bloom_rcp45 <- within(bloom_rcp45, remove(location, latitude, longitude, value, month, day))

setnames(bloom_rcp85, old=c("ClimateGroup"), new=c("time_period"))
setnames(bloom_rcp45, old=c("ClimateGroup"), new=c("time_period"))

setnames(bloom_rcp85, old=c("ClimateScenario"), new=c("model"))
setnames(bloom_rcp45, old=c("ClimateScenario"), new=c("model"))

bloom_rcp85$model <- as.character(bloom_rcp85$model)
bloom_rcp85$model[bloom_rcp85$model=="historical"] <- "observed"

bloom_rcp45$model <- as.character(bloom_rcp45$model)
bloom_rcp45$model[bloom_rcp45$model=="historical"] <- "observed"

write_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/"

saveRDS(bloom_rcp45, paste0(write_dir, "bloom_cloudy_45_50Percent.rds"))
saveRDS(bloom_rcp85, paste0(write_dir, "bloom_cloudy_85_50Percent.rds"))


