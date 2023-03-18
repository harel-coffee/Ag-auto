#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir= "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/quantiles_data/"
name_pref = "larva_data"

args = commandArgs(trailingOnly=TRUE)
quan = as.double(args[1])
quan = 0.5

min_pop_cut_off <- c(0.005, 0.01, 0.02, 0.04, 0.05)
min_pop_cut_off = c(0.01)
models <- c("45.rds", "85.rds")
for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_pref, model)))
	curr_data$latitude = as.character(curr_data$latitude)
	curr_data$longitude= as.character(curr_data$longitude)

	gen_1 <- subset(curr_data, select = c(ClimateGroup, CountyGroup, dayofyear, year, PercLarvaGen1, latitude, longitude))
	gen_2 <- subset(curr_data, select = c(ClimateGroup, CountyGroup, dayofyear, year, PercLarvaGen2, latitude, longitude))
	gen_3 <- subset(curr_data, select = c(ClimateGroup, CountyGroup, dayofyear, year, PercLarvaGen3, latitude, longitude))
	gen_4 <- subset(curr_data, select = c(ClimateGroup, CountyGroup, dayofyear, year, PercLarvaGen4, latitude, longitude))
	rm(curr_data) # free up memory
	for (cut in min_pop_cut_off){
		gen_1_cut <- gen_1[gen_1$PercLarvaGen1 >= cut, ]
		gen_2_cut <- gen_2[gen_2$PercLarvaGen2 >= cut, ]
		gen_3_cut <- gen_3[gen_3$PercLarvaGen3 >= cut, ]
		gen_4_cut <- gen_4[gen_4$PercLarvaGen4 >= cut, ]
		saveRDS(gen_1_cut, paste0(output_dir, "gen_1_cut_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(gen_2_cut, paste0(output_dir, "gen_2_cut_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(gen_3_cut, paste0(output_dir, "gen_3_cut_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(gen_4_cut, paste0(output_dir, "gen_4_cut_", as.character(quan*100), "_", as.character(cut), "_", model))

		start_wind_gen_1 <- gen_1_cut %>% group_by(ClimateGroup, CountyGroup,latitude, longitude, year) %>% arrange(PercLarvaGen1) %>% slice(1)
		start_wind_gen_2 <- gen_2_cut %>% group_by(ClimateGroup, CountyGroup,latitude, longitude, year) %>% arrange(PercLarvaGen2) %>% slice(1)
		start_wind_gen_3 <- gen_3_cut %>% group_by(ClimateGroup, CountyGroup,latitude, longitude, year) %>% arrange(PercLarvaGen3) %>% slice(1)
		start_wind_gen_4 <- gen_4_cut %>% group_by(ClimateGroup, CountyGroup,latitude, longitude, year) %>% arrange(PercLarvaGen4) %>% slice(1)

		names(start_wind_gen_1)[names(start_wind_gen_1) == "dayofyear"] = "window_start"
		names(start_wind_gen_2)[names(start_wind_gen_2) == "dayofyear"] = "window_start"
		names(start_wind_gen_3)[names(start_wind_gen_3) == "dayofyear"] = "window_start"
		names(start_wind_gen_4)[names(start_wind_gen_4) == "dayofyear"] = "window_start"
		saveRDS(start_wind_gen_1, paste0(output_dir, "start_wind_gen_1_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(start_wind_gen_2, paste0(output_dir, "start_wind_gen_2_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(start_wind_gen_3, paste0(output_dir, "start_wind_gen_3_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(start_wind_gen_4, paste0(output_dir, "start_wind_gen_4_", as.character(quan*100), "_", as.character(cut), "_", model))

		end_wind_gen_1 = gen_1_cut %>% group_by(ClimateGroup, CountyGroup, latitude, longitude, year) %>% arrange(abs(PercLarvaGen1 - quan)) %>% slice(1)
		end_wind_gen_2 = gen_2_cut %>% group_by(ClimateGroup, CountyGroup, latitude, longitude, year) %>% arrange(abs(PercLarvaGen2 - quan)) %>% slice(1)
		end_wind_gen_3 = gen_3_cut %>% group_by(ClimateGroup, CountyGroup, latitude, longitude, year) %>% arrange(abs(PercLarvaGen3 - quan)) %>% slice(1)
		end_wind_gen_4 = gen_4_cut %>% group_by(ClimateGroup, CountyGroup, latitude, longitude, year) %>% arrange(abs(PercLarvaGen4 - quan)) %>% slice(1)

		names(end_wind_gen_1)[names(end_wind_gen_1) == "dayofyear"] = "window_end"
		names(end_wind_gen_2)[names(end_wind_gen_2) == "dayofyear"] = "window_end"
		names(end_wind_gen_3)[names(end_wind_gen_3) == "dayofyear"] = "window_end"
		names(end_wind_gen_4)[names(end_wind_gen_4) == "dayofyear"] = "window_end"
		saveRDS(end_wind_gen_1, paste0(output_dir, "end_wind_gen_1_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(end_wind_gen_2, paste0(output_dir, "end_wind_gen_2_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(end_wind_gen_3, paste0(output_dir, "end_wind_gen_3_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(end_wind_gen_4, paste0(output_dir, "end_wind_gen_4_", as.character(quan*100), "_", as.character(cut), "_", model))

		merged_gen_1 = merge(x = start_wind_gen_1, y = end_wind_gen_1, by=c("ClimateGroup", "CountyGroup", "year", "latitude", "longitude"))
		merged_gen_2 = merge(x = start_wind_gen_2, y = end_wind_gen_2, by=c("ClimateGroup", "CountyGroup", "year", "latitude", "longitude"))
		merged_gen_3 = merge(x = start_wind_gen_3, y = end_wind_gen_3, by=c("ClimateGroup", "CountyGroup", "year", "latitude", "longitude"))
		merged_gen_4 = merge(x = start_wind_gen_4, y = end_wind_gen_4, by=c("ClimateGroup", "CountyGroup", "year", "latitude", "longitude"))

		saveRDS(merged_gen_1, paste0(output_dir, "merged_gen_1_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(merged_gen_2, paste0(output_dir, "merged_gen_2_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(merged_gen_3, paste0(output_dir, "merged_gen_3_", as.character(quan*100), "_", as.character(cut), "_", model))
		saveRDS(merged_gen_4, paste0(output_dir, "merged_gen_4_", as.character(quan*100), "_", as.character(cut), "_", model))

		merged_gen_1$window_gen_1 = merged_gen_1$window_end - merged_gen_1$window_start
		merged_gen_2$window_gen_2 = merged_gen_2$window_end - merged_gen_2$window_start
		merged_gen_3$window_gen_3 = merged_gen_3$window_end - merged_gen_3$window_start
		merged_gen_4$window_gen_4 = merged_gen_4$window_end - merged_gen_4$window_start


		
		#saveRDS(merged_gen_1, paste0(output_dir, "Larva_gen_1_", as.character(quan*100), "_", as.character(cut), "_", model))
		#saveRDS(merged_gen_2, paste0(output_dir, "Larva_gen_2_", as.character(quan*100), "_", as.character(cut), "_", model))
		#saveRDS(merged_gen_3, paste0(output_dir, "Larva_gen_3_", as.character(quan*100), "_", as.character(cut), "_", model))
		#saveRDS(merged_gen_4, paste0(output_dir, "Larva_gen_4_", as.character(quan*100), "_", as.character(cut), "_", model))

		#merged_gen_1 <- subset(merged_gen_1, select=c("ClimateGroup", "CountyGroup", "window_gen_1"))
		#merged_gen_2 <- subset(merged_gen_2, select=c("ClimateGroup", "CountyGroup", "window_gen_2"))
		#merged_gen_3 <- subset(merged_gen_3, select=c("ClimateGroup", "CountyGroup", "window_gen_3"))
		#merged_gen_4 <- subset(merged_gen_4, select=c("ClimateGroup", "CountyGroup", "window_gen_4"))

		#merged_gen_1 <- melt(merged_gen_1, id=c("ClimateGroup", "CountyGroup"))
		#merged_gen_2 <- melt(merged_gen_2, id=c("ClimateGroup", "CountyGroup"))
		#merged_gen_3 <- melt(merged_gen_3, id=c("ClimateGroup", "CountyGroup"))
		#merged_gen_4 <- melt(merged_gen_4, id=c("ClimateGroup", "CountyGroup"))
        
        #data <- rbind(merged_gen_1, merged_gen_2, merged_gen_3, merged_gen_4)
        #names(data)[names(data) == "variable"] = "generations"
        #names(data)[names(data) == "value"] = "window_length"

        #data$CountyGroup = as.character(data$CountyGroup)
        #data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
        #data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

		#saveRDS(data, paste0(output_dir, "Larva_", as.character(quan*100), "_", as.character(cut), "_", model))
	}
}