#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_4_pest_rcp"
#models = c("45.rds")
#stages = ("Larva", "Adult")

args = commandArgs(trailingOnly=TRUE)
models = args[1]
stages = args[2]

for (model in models){
	for (stage in stages){
		df_25 <- data.frame(matrix(ncol = 8, nrow = 0))
		df_50 <- data.frame(matrix(ncol = 8, nrow = 0))
		df_75 <- data.frame(matrix(ncol = 8, nrow = 0))
		df_100 <- data.frame(matrix(ncol = 8, nrow = 0))
		col_pref = c(paste0("Perc", stage, "Gen1"),
			         paste0("Perc", stage, "Gen2"),
			         paste0("Perc", stage, "Gen3"),
			         paste0("Perc", stage, "Gen4"))
		
		L <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", col_pref)

		colnames(df_25) <- L
		colnames(df_50) <- L
		colnames(df_75) <- L
		colnames(df_100)<- L

		curr_data = readRDS(paste0(data_dir, name_pref, model))

		n_rows = dim(curr_data)[1]
		if (stage == "Larva"){
			curr_data = subset(curr_data, select = c(ClimateGroup, CountyGroup, CumDDinF, dayofyear,
				                                     PercLarvaGen1, PercLarvaGen2, 
		                                             PercLarvaGen3, PercLarvaGen4))
			for (row_count in 2:n_rows){
				###################################          Gen 1            ############################
				## 25%
				if (curr_data$PercLarvaGen1[row_count]>0.25 & curr_data$PercLarvaGen1[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercLarvaGen1[row_count]>0.5 & curr_data$PercLarvaGen1[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercLarvaGen1[row_count]>0.75 & curr_data$PercLarvaGen1[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
			    ## 100%
				if (curr_data$PercLarvaGen1[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
				###################################          Gen 2            ############################
				## 25%
				if (curr_data$PercLarvaGen2[row_count]>0.25 & curr_data$PercLarvaGen2[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercLarvaGen2[row_count]>0.5 & curr_data$PercLarvaGen2[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercLarvaGen2[row_count]>0.75 & curr_data$PercLarvaGen2[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
				## 100%
				if (curr_data$PercLarvaGen2[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
				###################################          Gen 3            ############################
				## 25%
				if (curr_data$PercLarvaGen3[row_count]>0.25 & curr_data$PercLarvaGen3[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercLarvaGen3[row_count]>0.5 & curr_data$PercLarvaGen3[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercLarvaGen3[row_count]>0.75 & curr_data$PercLarvaGen3[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
				## 100%
				if (curr_data$PercLarvaGen3[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
				###################################          Gen 4            ############################
				## 25%
				if (curr_data$PercLarvaGen4[row_count]>0.25 & curr_data$PercLarvaGen4[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}

				## 50%
				if (curr_data$PercLarvaGen4[row_count]>0.5 & curr_data$PercLarvaGen4[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercLarvaGen4[row_count]>0.75 & curr_data$PercLarvaGen4[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
				## 100%
				if (curr_data$PercLarvaGen4[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
			}
		saveRDS(df_25, paste0(output_dir,  "pest_quantile",  "25_", stage,"rcp", model))
        saveRDS(df_50, paste0(output_dir,  "pest_quantile",  "50_", stage,"rcp", model))
        saveRDS(df_75, paste0(output_dir,  "pest_quantile",  "75_", stage,"rcp", model))
        saveRDS(df_100, paste0(output_dir, "pest_quantile", "100_", stage,"rcp", model))
		}
		##############################################################################################
		############################			                    ##################################
		############################               Adult            ##################################
		############################			                    ##################################
		##############################################################################################
		if (stage == "Adult"){
			curr_data = subset(curr_data, select = c(ClimateGroup, CountyGroup, CumDDinF, dayofyear,
				                                     PercAdultGen1, PercAdultGen2, 
				                                     PercAdultGen3, PercAdultGen4))
			for (row_count in 2:n_rows){
			    #################################          Gen 1            ##################################
		        ## 25%
				if (curr_data$PercAdultGen1[row_count]>0.25 & curr_data$PercAdultGen1[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercAdultGen1[row_count]>0.5 & curr_data$PercAdultGen1[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercAdultGen1[row_count]>0.75 & curr_data$PercAdultGen1[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
			    ## 100%
				if (curr_data$PercAdultGen1[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
				###################################          Gen 2            ############################
				## 25%
				if (curr_data$PercAdultGen2[row_count]>0.25 & curr_data$PercAdultGen2[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercAdultGen2[row_count]>0.5 & curr_data$PercAdultGen2[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}

				## 75%
				if (curr_data$PercAdultGen2[row_count]>0.75 & curr_data$PercAdultGen2[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}

				## 100%
				if (curr_data$PercAdultGen2[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
	            ###################################          Gen 3            ############################
				## 25%
				if (curr_data$PercAdultGen3[row_count]>0.25 & curr_data$PercAdultGen3[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercAdultGen3[row_count]>0.5 & curr_data$PercAdultGen3[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercAdultGen3[row_count]>0.75 & curr_data$PercAdultGen3[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
				## 100%
				if (curr_data$PercAdultGen3[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
				###################################          Gen 4            ############################
				## 25%
				if (curr_data$PercAdultGen4[row_count]>0.25 & curr_data$PercAdultGen4[row_count-1]<0.25){
					df_25 = rbind(df_25, curr_data[row_count, ])
				}
				## 50%
				if (curr_data$PercAdultGen4[row_count]>0.5 & curr_data$PercAdultGen4[row_count-1]<0.5){
					df_50 = rbind(df_50, curr_data[row_count, ])
				}
				## 75%
				if (curr_data$PercAdultGen4[row_count]>0.75 & curr_data$PercAdultGen4[row_count-1]<0.75){
					df_75 = rbind(df_75, curr_data[row_count, ])
				}
				## 100%
				if (curr_data$PercAdultGen4[row_count]>0.99){
					df_100 = rbind(df_100, curr_data[row_count, ])
				}
			}
		}
    saveRDS(df_25, paste0(output_dir,  "pest_quantile",  "25_", stage,"rcp", model))
    saveRDS(df_50, paste0(output_dir,  "pest_quantile",  "50_", stage,"rcp", model))
    saveRDS(df_75, paste0(output_dir,  "pest_quantile",  "75_", stage,"rcp", model))
    saveRDS(df_100, paste0(output_dir, "pest_quantile", "100_", stage,"rcp", model))
	}
}
