#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_4_pest_rcp"
models = c("45.rds")
models = c("85.rds")

for (model in models){
	df_25 <- data.frame(matrix(ncol = 12, nrow = 0))
	df_50 <- data.frame(matrix(ncol = 12, nrow = 0))
	df_75 <- data.frame(matrix(ncol = 12, nrow = 0))
	df_100 <- data.frame(matrix(ncol = 12, nrow = 0))
	colnames(df_25) <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
		                 "PercLarvaGen1", "PercLarvaGen2", 
		                 "PercLarvaGen3", "PercLarvaGen4",
		                 "PercAdultGen1", "PercAdultGen2", 
		                 "PercAdultGen3", "PercAdultGen4")

	colnames(df_50) <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
   		                 "PercLarvaGen1", "PercLarvaGen2", 
		                 "PercLarvaGen3", "PercLarvaGen4",
		                 "PercAdultGen1", "PercAdultGen2", 
		                 "PercAdultGen3", "PercAdultGen4")

	colnames(df_75) <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
		                 "PercLarvaGen1", "PercLarvaGen2", 
		                 "PercLarvaGen3", "PercLarvaGen4",
		                 "PercAdultGen1", "PercAdultGen2", 
		                 "PercAdultGen3", "PercAdultGen4")

	colnames(df_100) <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
		                  "PercLarvaGen1", "PercLarvaGen2", 
		                  "PercLarvaGen3", "PercLarvaGen4",
		                  "PercAdultGen1", "PercAdultGen2", 
		                  "PercAdultGen3", "PercAdultGen4")

	curr_data = readRDS(paste0(data_dir, name_pref, model))

	n_rows = dim(curr_data)[1]

	for (row_count in 2:n_rows){

		###################################          Gen 1            ############################
		############               ############
		############     Larva     ############
		############			   ############
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

        #######################################
		############     Adult     ############
		############			   ############
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
		############               ############
		############     Larva     ############
		############			   ############
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
		#######################################
		############     Adult     ############
		############			   ############
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
		############               ############
		############     Larva     ############
		############			   ############
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
		#######################################
		############     Adult     ############
		############			   ############
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
		#######################################
		############     Larva     ############
		############			   ############
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
		############               ############
		############     Adult     ############
		############			   ############
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
    df <- rbind(df_25, df_50, df_75)
    output_name = paste0("pest_quantile_rcp", model)
    saveRDS(df, paste0(output_dir, output_name))
}
