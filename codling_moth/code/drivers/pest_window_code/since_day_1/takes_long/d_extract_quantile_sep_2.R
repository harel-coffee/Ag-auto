#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_4_pest_rcp"

args = commandArgs(trailingOnly=TRUE)
model = args[1]

df_h <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(df_h) <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
	                 "PercLarvaGen1", "PercLarvaGen2", 
	                 "PercLarvaGen3", "PercLarvaGen4",
	                 "PercAdultGen1", "PercAdultGen2", 
	                 "PercAdultGen3", "PercAdultGen4")

df_larva_25 <- df_h
df_larva_50 <- df_h
df_larva_75 <- df_h
df_larva_100 <- df_h

df_adult_25 <- df_h
df_adult_50 <- df_h
df_adult_75 <- df_h
df_adult_100 <- df_h
rm(df_h)

curr_data = readRDS(paste0(data_dir, name_pref, model))

n_rows = dim(curr_data)[1]
for (row_count in 2:n_rows){

	###################################          Gen 1            ############################
	############               ############
	############     Larva     ############
	############			   ############
	## 25%
	if (curr_data$PercLarvaGen1[row_count]>0.25 & curr_data$PercLarvaGen1[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercLarvaGen1[row_count]>0.5 & curr_data$PercLarvaGen1[row_count-1]<0.5){
		df_larva_50 = rbind(df_larva_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercLarvaGen1[row_count]>0.75 & curr_data$PercLarvaGen1[row_count-1]<0.75){
		df_larva_75 = rbind(df_larva_75, curr_data[row_count, ])
	}

    ## 100%
	if (curr_data$PercLarvaGen1[row_count]>0.99){
		df_larva_100 = rbind(df_larva_100, curr_data[row_count, ])
	}

    #######################################
	############     Adult     ############
	############			   ############
	## 25%
	if (curr_data$PercAdultGen1[row_count]>0.25 & curr_data$PercAdultGen1[row_count-1]<0.25){
		df_adult_25 = rbind(df_adult_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercAdultGen1[row_count]>0.5 & curr_data$PercAdultGen1[row_count-1]<0.5){
		df_adult_50 = rbind(df_adult_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercAdultGen1[row_count]>0.75 & curr_data$PercAdultGen1[row_count-1]<0.75){
		df_adult_75 = rbind(df_adult_75, curr_data[row_count, ])
	}


    ## 100%
	if (curr_data$PercAdultGen1[row_count]>0.99){
		df_adult_100 = rbind(df_adult_100, curr_data[row_count, ])
	}

	###################################          Gen 2            ############################
	############               ############
	############     Larva     ############
	############			   ############
	## 25%
	if (curr_data$PercLarvaGen2[row_count]>0.25 & curr_data$PercLarvaGen2[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercLarvaGen2[row_count]>0.5 & curr_data$PercLarvaGen2[row_count-1]<0.5){
		df_larva_50 = rbind(df_larva_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercLarvaGen2[row_count]>0.75 & curr_data$PercLarvaGen2[row_count-1]<0.75){
		df_larva_75 = rbind(df_larva_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercLarvaGen2[row_count]>0.99){
		df_larva_100 = rbind(df_larva_100, curr_data[row_count, ])
	}
	#######################################
	############     Adult     ############
	############			   ############
	## 25%
	if (curr_data$PercAdultGen2[row_count]>0.25 & curr_data$PercAdultGen2[row_count-1]<0.25){
		df_adult_25 = rbind(df_adult_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercAdultGen2[row_count]>0.5 & curr_data$PercAdultGen2[row_count-1]<0.5){
		df_adult_50 = rbind(df_adult_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercAdultGen2[row_count]>0.75 & curr_data$PercAdultGen2[row_count-1]<0.75){
		df_adult_75 = rbind(df_adult_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercAdultGen2[row_count]>0.99){
		df_adult_100 = rbind(df_adult_100, curr_data[row_count, ])
	}

	###################################          Gen 3            ############################
	############               ############
	############     Larva     ############
	############			   ############
	## 25%
	if (curr_data$PercLarvaGen3[row_count]>0.25 & curr_data$PercLarvaGen3[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercLarvaGen3[row_count]>0.5 & curr_data$PercLarvaGen3[row_count-1]<0.5){
		df_larva_50 = rbind(df_larva_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercLarvaGen3[row_count]>0.75 & curr_data$PercLarvaGen3[row_count-1]<0.75){
		df_larva_75 = rbind(df_larva_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercLarvaGen3[row_count]>0.99){
		df_larva_100 = rbind(df_larva_100, curr_data[row_count, ])
	}
	#######################################
	############     Adult     ############
	############			   ############
	## 25%
	if (curr_data$PercAdultGen3[row_count]>0.25 & curr_data$PercAdultGen3[row_count-1]<0.25){
		df_adult_25 = rbind(df_adult_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercAdultGen3[row_count]>0.5 & curr_data$PercAdultGen3[row_count-1]<0.5){
		df_adult_50 = rbind(df_adult_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercAdultGen3[row_count]>0.75 & curr_data$PercAdultGen3[row_count-1]<0.75){
		df_adult_75 = rbind(df_adult_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercAdultGen3[row_count]>0.99){
		df_adult_100 = rbind(df_adult_100, curr_data[row_count, ])
	}

	###################################          Gen 4            ############################
	#######################################
	############     Larva     ############
	############			   ############
	## 25%
	if (curr_data$PercLarvaGen4[row_count]>0.25 & curr_data$PercLarvaGen4[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercLarvaGen4[row_count]>0.5 & curr_data$PercLarvaGen4[row_count-1]<0.5){
		df_larva_50 = rbind(df_larva_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercLarvaGen4[row_count]>0.75 & curr_data$PercLarvaGen4[row_count-1]<0.75){
		df_larva_75 = rbind(df_larva_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercLarvaGen4[row_count]>0.99){
		df_larva_100 = rbind(df_larva_100, curr_data[row_count, ])
	}
	############               ############
	############     Adult     ############
	############			   ############
	## 25%
	if (curr_data$PercAdultGen4[row_count]>0.25 & curr_data$PercAdultGen4[row_count-1]<0.25){
		df_adult_25 = rbind(df_adult_25, curr_data[row_count, ])
	}

	## 50%
	if (curr_data$PercAdultGen4[row_count]>0.5 & curr_data$PercAdultGen4[row_count-1]<0.5){
		df_adult_50 = rbind(df_adult_50, curr_data[row_count, ])
	}

	## 75%
	if (curr_data$PercAdultGen4[row_count]>0.75 & curr_data$PercAdultGen4[row_count-1]<0.75){
		df_adult_75 = rbind(df_adult_75, curr_data[row_count, ])
	}

	## 100%
	if (curr_data$PercAdultGen4[row_count]>0.99){
		df_adult_100 = rbind(df_adult_100, curr_data[row_count, ])
	}
}


saveRDS(df_adult_25, paste0(output_dir, "df_adult_25_sep_2.rds"))
saveRDS(df_adult_50, paste0(output_dir, "df_adult_50_sep_2.rds"))
saveRDS(df_adult_75, paste0(output_dir, "df_adult_75_sep_2.rds"))
saveRDS(df_adult_100, paste0(output_dir, "df_adult_100_sep_2.rds"))

saveRDS(df_larva_25, paste0(output_dir, "df_larva_25_sep_2.rds"))
saveRDS(df_larva_50, paste0(output_dir, "df_larva_50_sep_2.rds"))
saveRDS(df_larva_75, paste0(output_dir, "df_larva_75_sep_2.rds"))
saveRDS(df_larva_100, paste0(output_dir, "df_larva_100_sep_2.rds"))





