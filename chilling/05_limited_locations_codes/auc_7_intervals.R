

rm(list=ls())

library(data.table)
library(dplyr)
library(MESS) # has the auc function in it.
library(zoo)
options(digits=9)

data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/7_time_intervals_data/"

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
             "(16, Inf)")
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)
month_names = c("Jan", "Feb", "Mar", "Sept", "Oct", "Nov", "Dec",
                "sept_thru_dec_modeled", "sept_thru_jan_modeled")

for (month in month_names){
    ##########################################
    #                                        #
    #         initialize the AUC table       #
    #         to populate                    #
    #                                        #
    ##########################################
    df_help <- data.frame(matrix(ncol = 15, nrow = 7))
    colnames(df_help) <- c("temp_intervals", "hist_warm", "hist_cold",
                           "rcp45_(2025_2050)_W", "rcp45_(2025_2050)_C",
                           "rcp45_(2051_2075)_W", "rcp45_(2051_2075)_C",
                           "rcp45_(2076_2099)_W", "rcp45_(2076_2099)_C",
                           "rcp85_(2025_2050)_W", "rcp85_(2025_2050)_C",
                           "rcp85_(2051_2075)_W", "rcp85_(2051_2075)_C",
                           "rcp85_(2076_2099)_W", "rcp85_(2076_2099)_C")

    df_help[, "temp_intervals"] = iof_char
    ##########################################
    #                                        #
    #         read the data off the disk     #
    #                                        #
    ##########################################

    data = data.table(readRDS(paste0(data_dir, month, ".rds")))
    data = within(data, remove(model, location, Month))

    data$ClimateGroup[data$Year <= 2005] <- "1950-2005"
    data$ClimateGroup[data$Year > 2025 & data$Year <= 2050] <- "2025-2050"
    data$ClimateGroup[data$Year > 2050 & data$Year <= 2075] <- "2051-2075"
    data$ClimateGroup[data$Year > 2075] <- "2076-2099"

    # There are years between (2005+1) and 2025 which ... becomes NA
    data = na.omit(data)

    # order the climate groups
    data$ClimateGroup <- factor(data$ClimateGroup, 
                                levels = c("1950-2005", "2025-2050", "2051-2075", "2076-2099"))

    ##########################################
    #                                        #
    #         separate scenarios             #
    #                                        #
    ##########################################
    data_hist = data %>% filter(scenario %in% c("historical"))
    data_45   = data %>% filter(scenario %in% c("rcp45"))
    data_85   = data %>% filter(scenario %in% c("rcp85"))
    rm(data)
   
    ##########################################
    #                                        #
    #          pick up proper years          #
    #                                        #
    ##########################################
    data_hist <- data_hist %>% 
                 filter(Year > 1950 & Year <= 2005,
                        Chill_season != "chill_1949-1950" &
                        Chill_season != "chill_2005-2006")

    data_45 <- data_45 %>% 
               filter(Year >= 2025 & Year <= 2100,
                      Chill_season != "chill_2025-2026")

    data_85 <- data_85 %>% 
               filter(Year >= 2025 & Year <= 2100,
                      Chill_season != "chill_2025-2026")

    ##########################################
    #                                        #
    #          separate time_periods         #
    #                  (ClimateGroup)        #
    #                                        #
    ########################################## 45
    data_45_2025_2050 <- data_45  %>% 
                         filter(Year >= 2025 & Year <= 2050,
                                Chill_season != "chill_2025-2026" &
                                Chill_season != "chill_2050-2051")

    data_45_2051_2075 <- data_45  %>% 
                         filter(Year > 2050 & Year <= 2075,
                                Chill_season != "chill_2050-2051" &
                                Chill_season != "chill_2075-2076")

    data_45_2076_2099 <- data_45  %>% 
                         filter(Year > 2075 & Year <= 2099,
                                Chill_season != "chill_2075-2076" &
                                Chill_season != "chill_2099-2100")

    ########################################### 85
    data_85_2025_2050 <- data_85  %>% 
                         filter(Year >= 2025 & Year <= 2050,
                                Chill_season != "chill_2025-2026" &
                                Chill_season != "chill_2050-2051")

    data_85_2051_2075 <- data_85  %>% 
                         filter(Year > 2050 & Year <= 2075,
                                Chill_season != "chill_2050-2051" &
                                Chill_season != "chill_2075-2076")

    data_85_2076_2099 <- data_85  %>% 
                         filter(Year > 2075 & Year <= 2099,
                                Chill_season != "chill_2075-2076" &
                                Chill_season != "chill_2099-2100")
    rm(data_45, data_85)

    ##########################################
    #                                        #
    #          separate warm/cool.           #
    #                                        #
    ########################################## H
    data_hist_w = data_hist %>% filter(CountyGroup == "Warmer Area")
    data_hist_c = data_hist %>% filter(CountyGroup == "Cooler Area")
    rm(data_hist)
    ########################################## 45
    data_45_2025_2050_w = data_45_2025_2050 %>% filter(CountyGroup == "Warmer Area")
    data_45_2051_2075_w = data_45_2051_2075 %>% filter(CountyGroup == "Warmer Area")
    data_45_2076_2099_w = data_45_2076_2099 %>% filter(CountyGroup == "Warmer Area")

    data_45_2025_2050_c = data_45_2025_2050 %>% filter(CountyGroup == "Cooler Area")
    data_45_2051_2075_c = data_45_2051_2075 %>% filter(CountyGroup == "Cooler Area")
    data_45_2076_2099_c = data_45_2076_2099 %>% filter(CountyGroup == "Cooler Area")
    rm(data_45_2025_2050, data_45_2051_2075, data_45_2076_2099)
    ########################################## 85
    data_85_2025_2050_w = data_85_2025_2050 %>% filter(CountyGroup == "Warmer Area")
    data_85_2051_2075_w = data_85_2051_2075 %>% filter(CountyGroup == "Warmer Area")
    data_85_2076_2099_w = data_85_2076_2099 %>% filter(CountyGroup == "Warmer Area")

    data_85_2025_2050_c = data_85_2025_2050 %>% filter(CountyGroup == "Cooler Area")
    data_85_2051_2075_c = data_85_2051_2075 %>% filter(CountyGroup == "Cooler Area")
    data_85_2076_2099_c = data_85_2076_2099 %>% filter(CountyGroup == "Cooler Area")
    rm(data_85_2025_2050, data_85_2051_2075, data_85_2076_2099)
    
    ##########################################
    #                                        #
    #          populate the AUC table        #
    #                                        #
    ##########################################    
    ##########################
    ## historical columns   ##
    ##########################
    ############
    #   warm   #
    ############

    X = data_hist_w$Temp
    Y = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "hist_warm"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "hist_warm"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "hist_warm"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "hist_warm"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "hist_warm"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "hist_warm"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "hist_warm"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    
    rm(X, Y, xt, yt, dt, data_hist_w)
    ############
    #   cold   #
    ############
    X = data_hist_c$Temp
    Y = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "hist_cold"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "hist_cold"] <- auc(x = (dt[xt > -2 & xt <= 4])$x,  y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "hist_cold"] <- auc(x = (dt[xt > 4  & xt <= 6])$x,  y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "hist_cold"] <- auc(x = (dt[xt > 6  & xt <= 8])$x,  y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "hist_cold"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y = (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "hist_cold"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y = (dt[xt > 13 & xt <= 16])$y)
    df_help[7, "hist_cold"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)

    rm(X, Y, xt, yt, dt, data_hist_c)
    
    ##########################
    ##    rcp45 columns     ##
    ##########################
    #################################################################################
    ###       (2025_2050)

    # rcp45_(2025_2050)_W
    X  = data_45_2025_2050_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2025_2050)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    
    rm(X, Y, xt, yt, dt, data_45_2025_2050_w)
    #######################
    # rcp45_(2025_2050)_C

    X  = data_45_2025_2050_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2025_2050)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    
    rm(X, Y, xt, yt, dt, data_45_2025_2050_c)
    #################################################################################
    ###       (2051_2075)

    X  = data_45_2051_2075_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2051_2075)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)

    rm(X, Y, xt, yt, dt, data_45_2051_2075_w)
    #########################################
    #### cold
    X  = data_45_2051_2075_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2051_2075)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_45_2051_2075_c)
    #################################################################################
    ## 2076_2099
    X  = data_45_2076_2099_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2076_2099)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    
    rm(X, Y, xt, yt, dt, data_45_2076_2099_w)
    ###################################################################
    #### cold
    X  = data_45_2076_2099_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp45_(2076_2099)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_45_2076_2099_c)

    ## rcp85 columns
    #################################################################################
    ###       (2025_2050)

    # rcp85_(2025_2050)_W

    X  = data_85_2025_2050_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2025_2050)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_85_2025_2050_w)

    # rcp85_(2025_2050)_C

    X  = data_85_2025_2050_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2025_2050)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    
    rm(X, Y, xt, yt, dt, data_85_2025_2050_c)
    #################################################################################
    ###       (2051_2075)

    X  = data_85_2051_2075_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2051_2075)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_85_2051_2075_w)
    
    #### cold
    X  = data_85_2051_2075_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2051_2075)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_85_2051_2075_c)
    #################################################################################
    X  = data_85_2076_2099_w$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > -2 & xt <= 4])$x, y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > 4  & xt <= 6])$x, y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > 6  & xt <= 8])$x, y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y= (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y=(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2076_2099)_W"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_85_2076_2099_w)
    
    #### cold
    X  = data_85_2076_2099_c$Temp
    Y  = density(X)
    xt = Y$x
    yt = Y$y
    dt = data.table(x = xt, y = yt)

    df_help[1, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt <= -2])$x, y = (dt[xt <= -2])$y)
    df_help[2, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > -2 & xt <= 4])$x,  y = (dt[xt > -2 & xt <= 4])$y)
    df_help[3, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > 4  & xt <= 6])$x,  y = (dt[xt > 4  & xt <= 6])$y)
    df_help[4, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > 6  & xt <= 8])$x,  y = (dt[xt > 6  & xt <= 8])$y)
    df_help[5, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > 8  & xt <= 13])$x, y = (dt[xt > 8  & xt <= 13])$y)
    df_help[6, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > 13 & xt <= 16])$x, y =(dt[xt > 13 & xt <= 16])$y)
    df_help[7, "rcp85_(2076_2099)_C"] <- auc(x = (dt[xt > 16])$x, y = (dt[xt > 16])$y)
    rm(X, Y, xt, yt, dt, data_85_2076_2099_c)
    
    write_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/7_time_intervals_data/"
    write.table(df_help, file = paste0(write_dir, month, "_auc.csv"),
                row.names=FALSE, 
                col.names=TRUE, 
                sep=",")

    ####################################################################################
    ########################################## Educational - For reference
    # file %>%
    # filter(Year > 2025 & Year <= 2055,
    # Chill_season != "chill_2025-2026" &
    # Chill_season != "chill_2055-2056") %>% 
    # group_by(Chill_season) %>%
    ##########################################
    
    # assign(x = paste0(month, "_density_plot_", "rcp45"),
    #        value = {plot_dens(data=data_45, month_name=month)})
    
    ##########################################
    # df[4, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)
    ####################################################################################
    
}

