escaped_diap_peaks_DDF <- function(file_path, file_name){
	file_name_read = paste0(file_path, file_name)
	data <- data.table(readRDS(file_name_read))
	data$CountyGroup = as.character(data$CountyGroup)
	data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
	data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
	
	data$variable <- factor(data$variable)
	if (substr(file_name, start=10, stop=12) == "abs"){
		data = data[data$variable=="AbsNonDiap", ]
		data <- data[variable =="AbsLarvaPop" | variable =="AbsNonDiap"]
	    data <- subset(data, select=c("ClimateGroup", "CountyGroup", 
		                              "CumulativeDDF", "variable", "value"))

	} else {
		data <- data[variable =="RelLarvaPop" | variable =="RelNonDiap"]
        data <- subset(data, select=c("ClimateGroup", "CountyGroup", 
        	                          "CumulativeDDF", "variable", "value"))
		data = data[data$variable=="RelNonDiap", ]
          }
	
	data_gen_1 <- data[data$CumulativeDDF >= 213  & data$CumulativeDDF < 1153,]
	data_gen_2 <- data[data$CumulativeDDF >= 1153 & data$CumulativeDDF < 2313,]
	data_gen_3 <- data[data$CumulativeDDF >= 2313 & data$CumulativeDDF < 3443,]
	data_gen_4 <- data[data$CumulativeDDF >= 3443 & data$CumulativeDDF < 4453,]

    data_gen_1_hist = data_gen_1[data_gen_1$ClimateGroup=="Historical"]
    data_gen_1_2040 = data_gen_1[data_gen_1$ClimateGroup=="2040's"]
    data_gen_1_2060 = data_gen_1[data_gen_1$ClimateGroup=="2060's"]
    data_gen_1_2080 = data_gen_1[data_gen_1$ClimateGroup=="2080's"]

    data_gen_2_hist = data_gen_2[data_gen_2$ClimateGroup=="Historical"]
    data_gen_2_2040 = data_gen_2[data_gen_2$ClimateGroup=="2040's"]
    data_gen_2_2060 = data_gen_2[data_gen_2$ClimateGroup=="2060's"]
    data_gen_2_2080 = data_gen_2[data_gen_2$ClimateGroup=="2080's"]

    data_gen_3_hist = data_gen_3[data_gen_3$ClimateGroup=="Historical"]
    data_gen_3_2040 = data_gen_3[data_gen_3$ClimateGroup=="2040's"]
    data_gen_3_2060 = data_gen_3[data_gen_3$ClimateGroup=="2060's"]
    data_gen_3_2080 = data_gen_3[data_gen_3$ClimateGroup=="2080's"]

    data_gen_4_hist = data_gen_4[data_gen_4$ClimateGroup=="Historical"]
    data_gen_4_2040 = data_gen_4[data_gen_4$ClimateGroup=="2040's"]
    data_gen_4_2060 = data_gen_4[data_gen_4$ClimateGroup=="2060's"]
    data_gen_4_2080 = data_gen_4[data_gen_4$ClimateGroup=="2080's"]

    DD_table = data.table(hist = c(0, 0, 0, 0),
    	                  "2040" = c(0, 0, 0, 0),
    	                  "2060" = c(0, 0, 0, 0),
    	                  "2080" = c(0, 0, 0, 0)
    	                  )

    DD_table[1, "hist"] = ifelse(max(data_gen_1_hist$value) == -Inf, -100, round(data_gen_1_hist[data_gen_1_hist$value == max(data_gen_1_hist$value), CumulativeDDF], 2))
    DD_table[2, "hist"] = ifelse(max(data_gen_2_hist$value) == -Inf, -100, round(data_gen_2_hist[data_gen_2_hist$value == max(data_gen_2_hist$value), CumulativeDDF], 2))
    DD_table[3, "hist"] = ifelse(max(data_gen_3_hist$value) == -Inf, -100, round(data_gen_3_hist[data_gen_3_hist$value == max(data_gen_3_hist$value), CumulativeDDF], 2))
    DD_table[4, "hist"] = ifelse(max(data_gen_4_hist$value) == -Inf, -100, round(data_gen_4_hist[data_gen_4_hist$value == max(data_gen_4_hist$value), CumulativeDDF], 2))

    DD_table[1, "2040"] = ifelse(max(data_gen_1_2040$value) == -Inf, -100, round(data_gen_1_2040[data_gen_1_2040$value == max(data_gen_1_2040$value), CumulativeDDF], 2))
    DD_table[2, "2040"] = ifelse(max(data_gen_2_2040$value) == -Inf, -100, round(data_gen_2_2040[data_gen_2_2040$value == max(data_gen_2_2040$value), CumulativeDDF], 2))
    DD_table[3, "2040"] = ifelse(max(data_gen_3_2040$value) == -Inf, -100, round(data_gen_3_2040[data_gen_3_2040$value == max(data_gen_3_2040$value), CumulativeDDF], 2))
    DD_table[4, "2040"] = ifelse(max(data_gen_4_2040$value) == -Inf, -100, round(data_gen_4_2040[data_gen_4_2040$value == max(data_gen_4_2040$value), CumulativeDDF], 2))

    DD_table[1, "2060"] = ifelse(max(data_gen_1_2060$value) == -Inf, -100, round(data_gen_1_2060[data_gen_1_2060$value == max(data_gen_1_2060$value), CumulativeDDF], 2))
    DD_table[2, "2060"] = ifelse(max(data_gen_1_2060$value) == -Inf, -100, round(data_gen_1_2060[data_gen_1_2060$value == max(data_gen_1_2060$value), CumulativeDDF], 2))
    DD_table[3, "2060"] = ifelse(max(data_gen_1_2060$value) == -Inf, -100, round(data_gen_1_2060[data_gen_1_2060$value == max(data_gen_1_2060$value), CumulativeDDF], 2))
    DD_table[4, "2060"] = ifelse(max(data_gen_1_2060$value) == -Inf, -100, round(data_gen_1_2060[data_gen_1_2060$value == max(data_gen_1_2060$value), CumulativeDDF], 2))

    DD_table[1, "2080"] = ifelse(max(data_gen_1_2080$value) == -Inf, -100, round(data_gen_1_2080[data_gen_1_2080$value == max(data_gen_1_2080$value), CumulativeDDF], 2))
    DD_table[2, "2080"] = ifelse(max(data_gen_1_2080$value) == -Inf, -100, round(data_gen_1_2080[data_gen_1_2080$value == max(data_gen_1_2080$value), CumulativeDDF], 2))
    DD_table[3, "2080"] = ifelse(max(data_gen_1_2080$value) == -Inf, -100, round(data_gen_1_2080[data_gen_1_2080$value == max(data_gen_1_2080$value), CumulativeDDF], 2))
    DD_table[4, "2080"] = ifelse(max(data_gen_1_2080$value) == -Inf, -100, round(data_gen_1_2080[data_gen_1_2080$value == max(data_gen_1_2080$value), CumulativeDDF], 2))
    write.table(DD_table, paste0(file_path, substr(file_name, start=1, stop=20), "_DD", ".csv"), 
                row.names = FALSE, col.names = TRUE, sep = ",")
    
}

