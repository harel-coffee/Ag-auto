.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)

options(digit=9)
options(digits=9)

################################################################################

setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/7_temp_int_limit_locs/")
setwd("/data/hydro/users/Hossein/chill/7_time_intervals/RDS_files/")

months = c("Sept.rds", "Oct.rds", "Nov.rds", "Dec.rds", "Jan.rds",
           "sept_thru_dec_modeled.rds", "sept_thru_jan_modeled.rds")

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

##########################################
#                                        #
#     initialize the AUC table           #
#     to populate                        #
#                                        #
##########################################
the_table = data.frame(matrix(ncol = 10, nrow = 0))
perc_table = data.frame(matrix(ncol = 10, nrow = 0))

for (month in months){
  ##########################################
  #                                        #
  #     read the data off the disk         #
  #                                        #
  ##########################################
  
  data <- data.table(readRDS(month))
  data <- data %>% filter(year <= 2005 | year > 2075) %>% data.table()
  data <- data %>% filter(scenario== "historical" | scenario== "rcp85") %>% data.table()
  data <- within(data, remove(model, month))

  ##########################################
  #                                        #
  #     separate scenarios                 #
  #                                        #
  ##########################################
  data_hist <- data %>% filter(scenario %in% c("historical")) %>% data.table()
  data_85   <- data %>% filter(scenario %in% c("rcp85")) %>% data.table()
  city_names <- unique(data$city)

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

  m = unlist(strsplit(month, "[.]"))[1]
  for (citi in city_names){
    
    # data table for actual counts
    df_help <- data.frame(matrix(ncol = 10, nrow = 3))
    colnames(df_help) <- c("city", "month", "time_period", iof_char)
    df_help[, "city"] = c(citi, citi, citi)
    df_help[, "month"] = c(m, m, m)
    df_help[, "time_period"] = c("hist", "76-99", "difference")

    df_help_perc <- df_help # data frame for percentages

    dt_int_hist <- data_hist %>% filter(city==citi)
    dt_int_F3 <- data_85_2076_2099 %>% filter(city==citi)

    df_help[1, 4:10] = table(cut(dt_int_hist$Temp, breaks = iof_breaks))
    df_help[2, 4:10] = table(cut(dt_int_F3$Temp, breaks = iof_breaks))
    df_help[3, 4:10] = df_help[2, 4:10] - df_help[1, 4:10]
    df_help[is.na(df_help)] <- 0

    the_table <- rbind(the_table, df_help)

    numeric_part <- df_help[1:3, 4:10]
    sum_rows <- rowSums(numeric_part)
    numeric_percentages_time <- numeric_part / abs(sum_rows)

    ####### We want the difference between actual percentages,
    # The division above messes up the differences, it does not map the actual
    # difference to actual percentage-difference. So, we have to do the following correction.
    numeric_percentages_time[3,] = numeric_percentages_time[2,] - numeric_percentages_time[1,]
    df_help_perc[1:3, 4:10] <- numeric_percentages_time

    perc_table <- rbind(perc_table, df_help_perc)
  }
}


perc_table$Four_to_13 = perc_table[, "(4, 6]"] + 
                        perc_table[, "(6, 8]"] + 
                        perc_table[, "(8, 13]"]

perc_table[, 4:11] = perc_table[, 4:11] * 100


write.table(x = perc_table, row.names=F, col.names = T, sep=",", file = "perc_table.csv")
write.table(x = the_table, row.names=F, col.names = T, sep=",", file =  "actual_counts.csv")
