#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_4_pest_rcp"
models = c("45.rds", "85.rds")

for (model in models){
	df <- data.frame(matrix(ncol = 8, nrow = 0))
	colnames(df) <- c("latitude", "longitude",
		              "ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
		              "PercLarvaGen1", "PercLarvaGen2", 
		              "PercLarvaGen3", "PercLarvaGen4")
	curr_data = readRDS(paste0(data_dir, name_pref, model))

	# Gen 1's
	L = curr_data[curr_data$PercLarvaGen1 > 0.25 ]
	L = L[L$PercLarvaGen1 == min(L$PercLarvaGen1)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen1 > 0.5 ]
	L = L[L$PercLarvaGen1 == min(L$PercLarvaGen1)]
	df = rbind(df, L)
	
	L = curr_data[curr_data$PercLarvaGen1 > 0.75 ]
	L = L[L$PercLarvaGen1 == min(L$PercLarvaGen1)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen1 > 0.999 ]
	L = L[L$PercLarvaGen1 == min(L$PercLarvaGen1)]
	df = rbind(df, L)


	# Gen 2's
	L = curr_data[curr_data$PercLarvaGen2 > 0.25 ]
	L = L[L$PercLarvaGen2 == min(L$PercLarvaGen2)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen2 > 0.5 ]
	L = L[L$PercLarvaGen2 == min(L$PercLarvaGen2)]
	df = rbind(df, L)
	
	L = curr_data[curr_data$PercLarvaGen2 > 0.75 ]
	L = L[L$PercLarvaGen2 == min(L$PercLarvaGen2)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen2 > 0.999 ]
	L = L[L$PercLarvaGen2 == min(L$PercLarvaGen2)]
	df = rbind(df, L)


	# Gen 3's
	L = curr_data[curr_data$PercLarvaGen3 > 0.25 ]
	L = L[L$PercLarvaGen3 == min(L$PercLarvaGen3)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen3 > 0.5 ]
	L = L[L$PercLarvaGen3 == min(L$PercLarvaGen3)]
	df = rbind(df, L)
	
	L = curr_data[curr_data$PercLarvaGen3 > 0.75 ]
	L = L[L$PercLarvaGen3 == min(L$PercLarvaGen3)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen3 > 0.999 ]
	L = L[L$PercLarvaGen3 == min(L$PercLarvaGen3)]
	df = rbind(df, L)

	# Gen 4's
	L = curr_data[curr_data$PercLarvaGen4 > 0.25 ]
	L = L[L$PercLarvaGen4 == min(L$PercLarvaGen4)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen4 > 0.5 ]
	L = L[L$PercLarvaGen4 == min(L$PercLarvaGen4)]
	df = rbind(df, L)
	
	L = curr_data[curr_data$PercLarvaGen4 > 0.75 ]
	L = L[L$PercLarvaGen4 == min(L$PercLarvaGen4)]
	df = rbind(df, L)

	L = curr_data[curr_data$PercLarvaGen4 > 0.999 ]
	L = L[L$PercLarvaGen4 == min(L$PercLarvaGen4)]
	df = rbind(df, L)

    output_name = paste0("pest_quantile_rcp", model)
    saveRDS(df, paste0(output_dir, output_name))
}
