#rm(list=ls())

library(MESS) # has the auc function in it.
library(data.table)
library(zoo)

data_dir = "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/diapause_sens/"

file_list = list.files(path = data_dir, 
                       pattern = ".rds", 
                       all.files = FALSE, 
                       full.names = FALSE, 
                       recursive = FALSE)

gen_borders = c(213, 1153, 2313, 3443, 4453)
time_periods = c("Historical", "2040's", "2060's", "2080's")
for (file in file_list){ 
  print (file)
  ######### initiate the output table
  pop_type = unlist(strsplit(file, "_"))[2]
  model_type = unlist(strsplit(unlist(strsplit(file, "_"))[3], ".rds"))
  df_help <- data.frame(matrix(ncol = 1, nrow = 4))
  colnames(df_help) <- c("help")
  df_help[1, 1] = "warmer_total"
  df_help[2, 1] = "warmer_escape"
  df_help[3, 1] = "colder_total"
  df_help[4, 1] = "colder_escape"

  ####### read data
  data <- data.table(readRDS(paste0(data_dir, file)))
  # subset the data to get just needed info.
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", "CumulativeDDF", "variable", "value"))
  
  for (time_p in time_periods){
    df <- data.frame(matrix(ncol = 5, nrow = 4))
    col_names = c("CountyGroup", 
                  paste0(time_p, "_gen_1"),
                  paste0(time_p, "_gen_2"),
                  paste0(time_p, "_gen_3"),
                  paste0(time_p, "_gen_4"))
    colnames(df) <- col_names
    df[1, 1] = "warmer_total"
    df[2, 1] = "warmer_escape"
    df[3, 1] = "colder_total"
    df[4, 1] = "colder_escape"
    
    if (time_p=="Historical"){
      curr_data <- data[data$ClimateGroup == "Historical"]
    } else if (time_p=="2040's"){
      curr_data <- data[data$ClimateGroup == "2040's"]
    } else if (time_p=="2060's"){
      curr_data <- data[data$ClimateGroup == "2060's"]
    } else if (time_p=="2080's"){
      curr_data <- data[data$ClimateGroup == "2080's"]
    }
    
    if (pop_type=="rel"){
      data_total <- curr_data[variable == "RelLarvaPop"]
      data_escap <- curr_data[variable == "RelNonDiap" ]
    } else {
      data_total <- curr_data[variable == "AbsLarvaPop"]
      data_escap <- curr_data[variable == "AbsNonDiap" ]
    }
    
    data_total_warm = data_total[data_total$CountyGroup==2]
    data_escap_warm = data_escap[data_escap$CountyGroup==2]
    
    data_total_cold = data_total[data_total$CountyGroup==1]
    data_escap_cold = data_escap[data_escap$CountyGroup==1]
    ##########################################
    ########################################## warmer total
    # gen 1
    data_gen = data_total_warm[data_total_warm$CumulativeDDF>=gen_borders[1] & data_total_warm$CumulativeDDF<gen_borders[2]]
    df[1, 2] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 2
    data_gen = data_total_warm[data_total_warm$CumulativeDDF>=gen_borders[2] & data_total_warm$CumulativeDDF<gen_borders[3]]
    df[1, 3] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 3
    data_gen = data_total_warm[data_total_warm$CumulativeDDF>=gen_borders[3] & data_total_warm$CumulativeDDF<gen_borders[4]]
    df[1, 4] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 4
    data_gen = data_total_warm[data_total_warm$CumulativeDDF>=gen_borders[4] & data_total_warm$CumulativeDDF<=gen_borders[5]]
    df[1, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    ##########################################
    ########################################## warmer escape
    # gen 1
    data_gen = data_escap_warm[data_escap_warm$CumulativeDDF>=gen_borders[1] & data_escap_warm$CumulativeDDF<gen_borders[2]]
    df[2, 2] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 2
    data_gen = data_escap_warm[data_escap_warm$CumulativeDDF>=gen_borders[2] & data_escap_warm$CumulativeDDF<gen_borders[3]]
    df[2, 3] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 3
    data_gen = data_escap_warm[data_escap_warm$CumulativeDDF>=gen_borders[3] & data_escap_warm$CumulativeDDF<gen_borders[4]]
    df[2, 4] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 4
    data_gen = data_escap_warm[data_escap_warm$CumulativeDDF>=gen_borders[4] & data_escap_warm$CumulativeDDF<=gen_borders[5]]
    df[2, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    ##########################################
    ########################################## colder total
    # gen 1
    data_gen = data_total_cold[data_total_cold$CumulativeDDF>=gen_borders[1] & data_total_cold$CumulativeDDF<gen_borders[2]]
    df[3, 2] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 2
    data_gen = data_total_cold[data_total_cold$CumulativeDDF>=gen_borders[2] & data_total_cold$CumulativeDDF<gen_borders[3]]
    df[3, 3] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 3
    data_gen = data_total_cold[data_total_cold$CumulativeDDF>=gen_borders[3] & data_total_cold$CumulativeDDF<gen_borders[4]]
    df[3, 4] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 4
    data_gen = data_total_cold[data_total_cold$CumulativeDDF>=gen_borders[4] & data_total_cold$CumulativeDDF<=gen_borders[5]]
    df[3, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    ##########################################
    ########################################## colder escape
    # gen 1
    data_gen = data_escap_cold[data_escap_cold$CumulativeDDF>=gen_borders[1] & data_escap_cold$CumulativeDDF<gen_borders[2]]
    df[4, 2] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 2
    data_gen = data_escap_cold[data_escap_cold$CumulativeDDF>=gen_borders[2] & data_escap_cold$CumulativeDDF<gen_borders[3]]
    df[4, 3] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 3
    data_gen = data_escap_cold[data_escap_cold$CumulativeDDF>=gen_borders[3] & data_escap_cold$CumulativeDDF<gen_borders[4]]
    df[4, 4] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    
    # gen 4
    data_gen = data_escap_cold[data_escap_cold$CumulativeDDF>=gen_borders[4] & data_escap_cold$CumulativeDDF<=gen_borders[5]]
    df[4, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    df_help <- cbind(df_help, df)
  }
  df_help <- data.frame(df_help)
  df_help <- within(df_help, remove(help))
  out_name = paste0(pop_type, "_", model_type, "_", 
                    substr(unlist(strsplit(file, "_"))[4], 1, 1), ".csv")
  print(substr(unlist(strsplit(file, "_"))[4], 1, 1))
  write.csv(df_help, paste0(data_dir, out_name), row.names = F)
}
