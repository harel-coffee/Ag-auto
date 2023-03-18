bloom_rcp45 <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/bloom_rcp45_0.5_new_TS_4_frostPlot.rds")

bloom_rcp85 <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/bloom_rcp85_0.5_new_TS_4_frostPlot.rds")

bloom_rcp45$emission <- "RCP 4.5"
bloom_rcp85$emission <- "RCP 8.5"

bloom <- rbind(bloom_rcp45, bloom_rcp85)

setnames(bloom, old=c("ClimateGroup"), new=c("time_period"))
bloom$location <- paste0(bloom$latitude, "_", bloom$longitude )
bloom <- within(bloom, remove(latitude, longitude))
bloom$time_period <- as.character(bloom$time_period)
bloom$time_period[bloom$time_period == "Historical"] <- "1979-2015"
saveRDS(bloom, "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/bloom_rcp45_50percent_TS.rds")

bloom_CTs <- read.csv("/Users/hn/Documents/GitHub/Ag/chilling/parameters/bloom_limited_cities.csv", as.is=TRUE)
bloom_CTs <- within(bloom_CTs, remove(lat, long))

bloom_limited <- bloom %>% filter(location %in% bloom_CTs$location)
bloom_limited <- merge(bloom_limited, bloom_CTs, all.x=TRUE)

saveRDS(bloom_limited, "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/bloom_limited_50Percent.rds")