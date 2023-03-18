#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_4_pest_rcp"

args = commandArgs(trailingOnly=TRUE)
model = args[1]

cols <- c("ClimateGroup", "CountyGroup", "CumDDinF", "dayofyear", 
	      "PercLarvaGen1", "PercLarvaGen2", 
	      "PercLarvaGen3", "PercLarvaGen4")

df_h <- data.frame(matrix(ncol = length(cols), nrow = 0))
colnames(df_h) <- cols
df_larva_25 <- df_h

rm(df_h)

curr_data = readRDS(paste0(data_dir, name_pref, model))
curr_data = subset(curr_data, select = c(ClimateGroup, CountyGroup, CumDDinF, dayofyear, 
		                                 PercLarvaGen1, PercLarvaGen2, 
		                                 PercLarvaGen3, PercLarvaGen4))
n_rows = dim(curr_data)[1]

for (row_count in 2:n_rows){
	###################################          Gen 1            ############################
	## 25%
	if (curr_data$PercLarvaGen1[row_count]>= 0.25 & curr_data$PercLarvaGen1[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}
	###################################          Gen 2            ############################
	## 25%
	if (curr_data$PercLarvaGen2[row_count]>= 0.25 & curr_data$PercLarvaGen2[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}
	##################################          Gen 3            ############################
	## 25%
	if (curr_data$PercLarvaGen3[row_count]>= 0.25 & curr_data$PercLarvaGen3[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}
	###################################          Gen 4            ############################
	## 25%
	if (curr_data$PercLarvaGen4[row_count]>=0.25 & curr_data$PercLarvaGen4[row_count-1]<0.25){
		df_larva_25 = rbind(df_larva_25, curr_data[row_count, ])
	}
}
saveRDS(df_larva_25, paste0(output_dir, "df_larva_25_", model))


