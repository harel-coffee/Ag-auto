rm(list=ls())
library(data.table)
library(dplyr)

setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/7_temp_int_limit_locs/")
months = c("Sept.rds", "Oct.rds", "Nov.rds", "Dec.rds", "Jan.rds",
           "sept_thru_dec_modeled.rds", "sept_thru_jan_modeled.rds")

months <- "Dec.rds"
iof = list(c(-Inf, -2), 
           c(-2, 4),
           c(4, 6),
           c(6, 8),
           c(8, 13),
           c(13, 16),
           c(16, Inf))
iof_char = c("(-Inf, -2]",
             "(-2, 4]",
             "(4, 6]",
             "(6, 8]",
             "(8, 13]",
             "(13, 16]",
             "(16, Inf]")
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

param_dir <- "/Users/hn/Documents/GitHub/Kirti/chilling/parameters/"
limited_locations <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), header=T, as.is=T))
city_names <- limited_locations$city

for (month in months){

  ##########################################
  #                                        #
  #     initialize the AUC table           #
  #     to populate                        #
  #                                        #
  ##########################################
  the_table = data.frame(matrix(ncol = 8, nrow = 0))
  df_help <- data.frame(matrix(ncol = 8, nrow = 15))
  colnames(df_help) <- c("city", iof_char)

  # The table to be filled with no. hours in each interval for each location
  # The time interval goes from sept. to Dec. or Jan. It is shortened
  # because variable names are getting long. It is NOT just for month of
  # Dec. or Jan.
  # The data for 2076-2099 are from RCP 8.5
  m = unlist(strsplit(month, "[.]"))[1]
  row_names <- paste0(m, c("_rich_hist", "_rich_76_99", "_rich_diff", 
                           "_omak_hist", "_omak_76_99", "_omak_diff",
                           "_Wenatchee_hist", "_Wenatchee_76_99", "_Wenatchee_diff",
                           "_Hilsboro_hist", "_Hilsboro_76_99", "_Hilsboro_diff", 
                           "_Elmira_hist", "_Elmira_76_99", "_Elmira_diff"))

  df_help[, "city"] = row_names
  ##########################################
  #                                        #
  #     read the data off the disk         #
  #                                        #
  ##########################################

  data = data.table(readRDS(month))
  data <- data %>% filter(year <= 2005 | year > 2075) %>% data.table()
  data <- data %>% filter(scenario== "historical" | scenario== "rcp85") %>% data.table()

  data[location=="46.28125_-119.34375"]$location <- "Richland"
  data[location=="48.40625_-119.53125"]$location <- "Omak"

  data[location=="47.40625_-120.34375"]$location <- "Wenatchee"
  data[location=="45.53125_-123.15625"]$location <- "Hillsboro"
  data[location=="44.09375_-123.34375"]$location <- "Elmira"

  data = within(data, remove(model, month))

  data$ClimateGroup[data$year <= 2005] <- "1950-2005"
  data$ClimateGroup[data$year >= 2076] <- "2076-2099"

  # order the climate groups
  data$ClimateGroup <- factor(data$ClimateGroup, levels = c("1950-2005", "2076-2099"))

  ##########################################
  #                                        #
  #     separate scenarios                 #
  #                                        #
  ##########################################
  data_hist <- data %>% filter(scenario %in% c("historical")) %>% data.table()
  data_85   <- data %>% filter(scenario %in% c("rcp85")) %>% data.table()

  rm(data)
  ##########################################
  #                                        #
  #      pick up proper years              #
  #                                        #
  ##########################################
  data_hist <- data_hist %>% 
               filter(year > 1950 & year <= 2005,
                      chill_season != "chill_1949-1950" &
                      chill_season != "chill_2005-2006")

  ########################################### 85
  data_85_2076_2099 <- data_85  %>% 
                       filter(year > 2075 & year <= 2099,
                              chill_season != "chill_2075-2076" &
                              chill_season != "chill_2099-2100")
  rm(data_85)
  ##########################################
  #                                        #
  #      separate cities                   #
  #                                        #
  ########################################## H
  data_hist_rich = data_hist %>% filter(location == "Richland")
  data_hist_omak = data_hist %>% filter(location == "Omak")
  
  data_hist_wenatchee = data_hist %>% filter(location == "Wenatchee")
  data_hist_hilsboro = data_hist %>% filter(location == "Hillsboro")
  data_hist_elmira = data_hist %>% filter(location == "Elmira")
  
  rm(data_hist)
  ########################################## 85
  data_85_2076_2099_rich = data_85_2076_2099 %>% filter(location == "Richland")
  data_85_2076_2099_omak = data_85_2076_2099 %>% filter(location == "Omak")

  data_85_2076_2099_wenatchee = data_85_2076_2099 %>% filter(location == "Wenatchee")
  data_85_2076_2099_hilsboro = data_85_2076_2099 %>% filter(location == "Hillsboro")
  data_85_2076_2099_elmira = data_85_2076_2099 %>% filter(location == "Elmira")
  
  rm(data_85_2076_2099)

  ##########################################
  #                                        #
  #      populate the table                #
  #                                        #
  ##########################################  
  ##########################################
  #   Richland   #
  ################
  #
  # v = data_hist_c %>% 
  #    mutate(temp_cat = cut(Temp, breaks = iof_breaks)) %>% 
  #    group_by(temp_cat) %>% 
  #    summarise(no_hours = n()) %>% data.table()
  
  df_help[1, 2:8] = table(cut(data_hist_rich$Temp, breaks = iof_breaks))
  df_help[2, 2:8] = table(cut(data_85_2076_2099_rich$Temp, breaks = iof_breaks))
  df_help[3, 2:8] = df_help[2, 2:8] - df_help[1, 2:8]

  ##########################################
  #   Omak       
  ################
  df_help[4, 2:8] = table(cut(data_hist_omak$Temp, breaks = iof_breaks))
  df_help[5, 2:8] = table(cut(data_85_2076_2099_omak$Temp, breaks = iof_breaks))
  df_help[6, 2:8] = df_help[5, 2:8] - df_help[4, 2:8]
  
  ##########################################
  #   Wenatchee    
  ################
  df_help[7, 2:8] = table(cut(data_hist_wenatchee$Temp, breaks = iof_breaks))
  df_help[8, 2:8] = table(cut(data_85_2076_2099_wenatchee$Temp, breaks = iof_breaks))
  df_help[9, 2:8] = df_help[8, 2:8] - df_help[7, 2:8]

  ##########################################
  #   Hillsboro    
  ################
  df_help[10, 2:8] = table(cut(data_hist_hilsboro$Temp, breaks = iof_breaks))
  df_help[11, 2:8] = table(cut(data_85_2076_2099_hilsboro$Temp, breaks = iof_breaks))
  df_help[12, 2:8] = df_help[11, 2:8] - df_help[10, 2:8]

  ##########################################
  #   Elmira    
  ################
  df_help[13, 2:8] = table(cut(data_hist_elmira$Temp, breaks = iof_breaks))
  df_help[14, 2:8] = table(cut(data_85_2076_2099_elmira$Temp, breaks = iof_breaks))
  df_help[15, 2:8] = df_help[14, 2:8] - df_help[13, 2:8]

  ####################################################################################
  the_table <- rbind(the_table, df_help)
  the_table[is.na(the_table)] <- 0

  actual_times <- the_table

  # drop the column name. It messes up the row-sum computation for example.
  numeric_actual_time <- actual_times[, 2:8]
  sum_rows <- rowSums(numeric_actual_time)
  numeric_percentages_time <- numeric_actual_time / abs(sum_rows)

  ####### We want the difference between actual percentages,
  # The divition above messes up the differences, it does not map the actual
  # difference to actual percentage-difference. So, we have to do the following correction.
  numeric_percentages_time[3,] = numeric_percentages_time[2,] - numeric_percentages_time[1,]
  numeric_percentages_time[6,] = numeric_percentages_time[5,] - numeric_percentages_time[4,]

  numeric_percentages_time$Four_to_13 = numeric_percentages_time[, 3] + 
                                        numeric_percentages_time[, 4] + 
                                        numeric_percentages_time[, 5]

  numeric_percentages_time = numeric_percentages_time * 100
  # put back the name column in there
  perc_table <- cbind(the_table[, 1], numeric_percentages_time)

  write.table(x = perc_table, row.names=F, col.names = T, sep=",",
              file = paste0("perc_table_", m, ".csv"))  
}

