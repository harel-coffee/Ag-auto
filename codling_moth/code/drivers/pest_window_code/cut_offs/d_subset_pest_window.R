#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_rcp"
models = c("45.rds", "85.rds")

for (model in models){
	curr_data = readRDS(paste0(data_dir, name_pref, model))
	curr_data = subset(curr_data, select = c(ClimateScenario, 
		                                     ClimateGroup, CountyGroup, 
		                                     CumDDinF, dayofyear, year,
		                                     PercLarvaGen1, PercLarvaGen2,
		                                     PercLarvaGen3, PercLarvaGen4,
		                                     latitude, longitude))
    output_name = paste0("larva_data_", model)
    saveRDS(curr_data, paste0(output_dir, output_name))
}

for (model in models){
	curr_data = readRDS(paste0(data_dir, name_pref, model))
	curr_data = subset(curr_data, select = c(ClimateScenario, ClimateGroup, CountyGroup, CumDDinF, dayofyear, year,
		                                     latitude, longitude,
		                                     PercAdultGen1, PercAdultGen2, 
		                                     PercAdultGen3, PercAdultGen4
		                                     ))
    output_name = paste0("adult_data_", model)
    saveRDS(curr_data, paste0(output_dir, output_name))
}