library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)



add_CH_DoY <- function(data){
  ################################
  ####
  #### Remove out-of-boundary years
  ####
  #
  # Toss unwanted data
  #
  data <- data.table(data)
  data$month <- month(data$Date)
  data <- data %>% filter(month %in% c(9, 10, 11, 12, 1, 2, 3, 4, 5))
  data <- data %>% filter(!(year == min(data$year) & month %in% c(1, 2, 3, 4, 5)))
  data <- data %>% filter(!(year == max(data$year) & month %in% c(9, 10, 11, 12)))
  
  data <- within(data, remove(month))
  data <- data.table(data)
  
  data <- data.table(data)
  data$CH_DoY <- 1 # dummy
  data[, CH_DoY := cumsum(CH_DoY), by=list(location, model, emission, hardiness_year)]
  data <- data.table(data)
  return(data)
}

hardiness_model <- function(data, 
                            input_params = input_params,
                            variety_params = variety_params){

    output = data.table(matrix(NA, nrow=dim(data)[1], ncol=12))
    colnames(output) <- c("variety", "location", "year",
                          "Date", "jday", "t_mean", "t_max", 
                          "t_min", "predicted_Hc", "observed_Hc",
                          "budbreak", "predicted_on")

    # output$predicted_on <- as.character(output$predicted_on)

    output$variety <- as.character(output$variety)
    output$location <- as.character(output$location)

    location = data$Location[1]
    year_1 = data$Season[1]
    # The following is just name of a given column
    # to be used to name a column in the output sheet/file.
    temp <- colnames(data)[8]
    variety <- input_params$variety[1]
    
    Hc_initial <- input_params$Hc_initial[1]
    Hc_min<- input_params$Hc_min[1]
    Hc_max <- input_params$Hc_max[1]
    ####################################################################
    #################
    ################# In the following vectors of size 2, the first
    ################# entry is _endo and the second is _eco type.
    #################
    ####################################################################
    T_threshold <- c(input_params$t_threshold_endo[1], 
                     input_params$t_threshold_eco[1])
    
    acclimation_rate <- c(input_params$acclimation_rate_endo[1],
                          input_params$acclimation_rate_eco[1])

    deacclimation_rate<- c(input_params$deacclimation_rate_endo[1],
                           input_params$deacclimation_rate_eco[1])
    ####################################################################
    ecodormancy_boundary <- input_params$Ecodormancy_boundary[1]
    theta <- c(1, input_params$theta[1])
    
    # calculate range of hardiness values possible, 
    # this is needed for the logistic component
    ################################################## What the hell is going on here?
    Hc_range = Hc_min - Hc_max 

    # initialize some of the parameters
    DD_heating_sum = 0
    DD_chilling_sum = 0
    base10_chilling_sum = 0
    model_Hc_yesterday = Hc_initial
    dormancy_period = 1

    # number of observations 
    n_rows = dim(data)[1]

    for (row_count in 1:n_rows){
           # jdate <- data[row_count, 1]
           # the following line is done so we can write
           # the result in CSV format. 
           # (when class of the variable was of format factor, it had problem)
           jdate = as.character((data$Date[row_count]))
        jday = data$jday[row_count]   # jday <- data[row_count, 3]
        # t_mean= data$T_mean[row_count] # t_mean = data[row_count, 5]
        
        if (is.na(data$T_mean[row_count])){
            message(sprintf("data$T_mean[row_count] is empty (NA) at row %s\n", row_count))
            break
        }
        t_max = data$T_max[row_count]    # t_max = data[row_count, 6]
        t_min = data$T_min[row_count]    # t_min = data[row_count, 7]
        observed_Hc = data$Observed_Hc[row_count] # observed_Hc = data[row_count, 8]

        # calculate heating degree days for today used in deacclimation
        if (data$T_mean[row_count] > T_threshold[dormancy_period]){
            DD_heating_today <- data$T_mean[row_count] - T_threshold[dormancy_period]
            } else {
                DD_heating_today = 0
            }
        
        # calculate cooling degree days for today used in acclimation
        if (data$T_mean[row_count] <= T_threshold[dormancy_period]){
            DD_chilling_today = data$T_mean[row_count] - T_threshold[dormancy_period]
            } else{
                DD_chilling_today = 0
            }

        # calculate cooling degree days using base 
        # of 10c to be used in dormancy release
        if(data$T_mean[row_count] <= 10){
            base10_chilling_today = data$T_mean[row_count] - 10
            } else {
                base10_chilling_today = 0
            }

        # calculate new model_Hc for today
        deacclimation = DD_heating_today * deacclimation_rate[dormancy_period] * 
                        (1 - ((model_Hc_yesterday - Hc_max) / Hc_range) ^ theta[[dormancy_period]])

        # do not allow deacclimation unless 
        # some chilling has occured, 
        # the actual start of the model
        if (DD_chilling_sum == 0){ deacclimation = 0}

        acclimation = DD_chilling_today * acclimation_rate[dormancy_period] * 
                      (1 - (Hc_min - model_Hc_yesterday) / Hc_range)
        Delta_Hc = acclimation + deacclimation
        model_Hc = model_Hc_yesterday + Delta_Hc

        # limit the hardiness to known min and max
        if (model_Hc <= Hc_max) {model_Hc = Hc_max}
        if (model_Hc > Hc_min) { model_Hc = Hc_min }

        # sum up chilling degree days
        DD_chilling_sum = DD_chilling_sum + DD_chilling_today

        base10_chilling_sum = base10_chilling_sum + base10_chilling_today

        # sum up heating degree days only if chilling requirement has been met
        #  i.e dormancy period 2 has started
        if (dormancy_period == 2) {DD_heating_sum = DD_heating_sum + DD_heating_today}

        # determine if chilling requirement has been met
        # re-set dormancy period
        # order of this and other if statements 
        # are consistant with Ferguson et al, or V6.3 of our SAS code
        if (base10_chilling_sum <= ecodormancy_boundary){dormancy_period = 2}

        output$variety[row_count] = as.character(variety)
        output$location[row_count] = as.character(location)
        output$year[row_count] = as.character(year_1)
        output$Date[row_count] = jdate
        output$jday[row_count] = jday
        output$t_mean[row_count] = data$T_mean[row_count]
        output$t_max[row_count] = t_max
        output$t_min[row_count] = t_min
        
        output$predicted_Hc[row_count] = model_Hc
        output$observed_Hc[row_count] = observed_Hc

        # use Hc_min to determine if vinifera or labrusca
        if (Hc_min == -1.2){
            if(model_Hc_yesterday < -2.2){
                if(model_Hc >= -2.2){
                    output$predicted_on[1] = jdate
                    output$budbreak[row_count] = model_Hc
                }
            }
        }

        if(Hc_min == -2.5){
            if(model_Hc_yesterday < -6.4){
                if(model_Hc >= -6.4){                    
                    output$predicted_on[1] = jdate
                    output$budbreak[row_count] = model_Hc
                }
            }
        }

        # remember todays hardiness for tomarrow
        model_Hc_yesterday = model_Hc
    }
    return(output)
}

