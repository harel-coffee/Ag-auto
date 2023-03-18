
mdata <- subset(mdata, select=c(chill_season, thresh_20:thresh_75, sum_A1, year, scenario, model, city))


mdata$time_period[mdata$year >= 2025 & mdata$year <= 2050] <- time_periods[2]
mdata$time_period[mdata$year >  2050 & mdata$year <= 2075] <- time_periods[3]
mdata$time_period[mdata$year >  2075] <- time_periods[4]
mdata$time_period = factor(mdata$time_period, levels=time_periods, order=T)


mdata$scenario[mdata$scenario == "rcp45"] <- "RCP 4.5"
mdata$scenario[mdata$scenario == "rcp85"] <- "RCP 8.5"
mdata$scenario[mdata$scenario == "historical"] <- "Historical"


dataT<- mdata

h_year_count <- length(unique(dataT[dataT$time_period =="Historical",]$chill_season))
f1_year_count <- length(unique(dataT[dataT$time_period== "2025_2050",]$chill_season))
f2_year_count <- length(unique(dataT[dataT$time_period== "2051_2075",]$chill_season))
f3_year_count <- length(unique(dataT[dataT$time_period== "2076_2099",]$chill_season))

if (due == "Jan"){
col_name = "sum_J1"
} else if (due == "Feb"){
  col_name = "sum_F1"
} else if(due =="Mar"){
  col_name = "sum_M1"
} else if(due =="Apr"){
  col_name = "sum_A1"
}

bks = c(-300, seq(20, 75, 5), 300)

result <- dataT %>%
        mutate(thresh_range = cut(get(col_name), breaks = bks )) %>%
        group_by(time_period, 
                 thresh_range, model, scenario, city) %>%
        summarize(no_years = n_distinct(chill_season)) %>% 
        data.table()

time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
result$time_period = factor(result$time_period, 
                          levels=time_periods,
                          order=T)

result_85 <- result %>% filter(scenario == "RCP 8.5")

result_85_2076_2099 <- result_85 %>% filter(time_period == "2076_2099")
result_85_2076_2099$thresh_range <- factor(result_85_2076_2099$thresh_range, order=T)
result_85_2076_2099$thresh_range <- fct_rev(result_85_2076_2099$thresh_range) %>% data.table()
result_85_2076_2099 <- result_85_2076_2099[order(thresh_range), ]

result_85_2076_2099_cumsum <- result_85_2076_2099 %>% 
                              group_by(time_period, model, scenario, city) %>% 
                              mutate(n_years_passed = cumsum(no_years)) %>% 
                              data.table()

result_hist <- result_85_2076_2099_cumsum %>% filter(time_period == "Historical") %>% data.table()
result_50 <- result_85_2076_2099_cumsum %>% filter(time_period == "2025_2050") %>% data.table()
result_75 <- result_85_2076_2099_cumsum %>% filter(time_period == "2051_2075") %>% data.table()
result_99 <- result_85_2076_2099_cumsum %>% filter(time_period == "2076_2099") %>% data.table()

result_hist$frac_passed = result_hist$n_years_passed / h_year_count
result_50$frac_passed = result_50$n_years_passed / f1_year_count
result_75$frac_passed = result_75$n_years_passed / f2_year_count
result_99$frac_passed = result_99$n_years_passed / f3_year_count



result$thresh_range <- factor(result$thresh_range, order=T)
result$thresh_range <- fct_rev(result$thresh_range)
result <- result[order(thresh_range), ]

result <- result %>% 
          group_by(time_period, model, scenario, city) %>% 
          mutate(n_years_passed = cumsum(no_years)) %>% 
          data.table()

result_hist <- result %>% filter(time_period == "Historical") %>% data.table()
result_50 <- result %>% filter(time_period == "2025_2050") %>% data.table()
result_75 <- result %>% filter(time_period == "2051_2075") %>% data.table()
result_99 <- result %>% filter(time_period == "2076_2099") %>% data.table()

result_hist$frac_passed = result_hist$n_years_passed / h_year_count
result_50$frac_passed = result_50$n_years_passed / f1_year_count
result_75$frac_passed = result_75$n_years_passed / f2_year_count
result_99$frac_passed = result_99$n_years_passed / f3_year_count

result <- rbind(result_hist, result_50, result_75, result_99)
result <- na.omit(result)