

bloom_rcp45_new <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/bloom_rcp45_0.5_new.rds")
bloom_rcp85_new <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/bloom_rcp85_0.5_new.rds")

bloom_rcp45_new$emission <- "RCP 4.5"
bloom_rcp85_new$emission <- "RCP 8.5"

bloom_rcp45_new <- within(bloom_rcp45_new, remove(County))
bloom_rcp85_new <- within(bloom_rcp85_new, remove(County))

setnames(bloom_rcp45_new, old=c("ClimateGroup"), new=c("time_period"))
setnames(bloom_rcp85_new, old=c("ClimateGroup"), new=c("time_period"))

bloom_50_percent <- rbind(bloom_rcp45_new, bloom_rcp85_new)
# change historical to 1979-2015
bloom_50_percent$time_period <- as.character(bloom_50_percent$time_period)
bloom_50_percent$time_period[bloom_50_percent$time_period == "Historical"] <- "1979-2015"
bloom_50_percent$location <- paste0(bloom_50_percent$latitude, "_", bloom_50_percent$longitude)

bloom_50_percent <- within(bloom_50_percent, remove(latitude, longitude))
saveRDS(bloom_50_percent, "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/bloom_50_percent.rds")

bloom_limited_cities <- read.csv("/Users/hn/Documents/GitHub/Ag/chilling/parameters/bloom_limited_cities.csv")
bloom_limited_cities <- within(bloom_limited_cities, remove(lat, long))

bloom_50_percent_limited <- bloom_50_percent %>% 
                            filter(location %in% bloom_limited_cities$location)

bloom_50_percent_limited <- merge(bloom_50_percent_limited, bloom_limited_cities,
                                  all.x=TRUE, by="location")

bloom_50_percent_limited <- within(bloom_50_percent_limited, remove(location))

saveRDS(bloom_50_percent_limited, 
       "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/bloom_50_percent_limited.rds")




