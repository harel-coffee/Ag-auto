
#### Remove out-of-boundary years
data_till_Jan <- data_till_Jan %>% filter(!(year == 1949 & month == 1))

data_till_Feb <- data_till_Feb %>% filter(!(year == 1949 & month == 1))
data_till_Feb <- data_till_Feb %>% filter(!(year == 1949 & month == 2))
#
# remove the turning point from historical to RCP stuff
#
data_till_Jan <- data_till_Jan %>% filter(year != 2005)
data_till_Feb <- data_till_Feb %>% filter(year != 2005)


if (due == "Jan"){
  first_frost <- first_frost %>% filter(!(year == 1949 & month == 1))
  fifth_frost <- fifth_frost %>% filter(!(year == 1949 & month == 1))
 
  first_frost <- first_frost %>% filter(year != 2005)
  fifth_frost <- fifth_frost %>% filter(year != 2005)

}

if (due == "Feb"){
  first_frost <- first_frost %>% filter(!(year == 1949 & month == 1))
  fifth_frost <- fifth_frost %>% filter(!(year == 1949 & month == 1))

  first_frost <- first_frost %>% filter(!(year == 1949 & month == 2))
  fifth_frost <- fifth_frost %>% filter(!(year == 1949 & month == 2))

  first_frost <- first_frost %>% filter(year != 2005)
  fifth_frost <- fifth_frost %>% filter(year != 2005)
}

form_chill_season_dayofyear <- function(dt, due){
  keycol <- c("location", "year", "month", "day")
  setorderv(dt, keycol)
  dt$dayofyear <- 1 # dummy
  dt[, dayofyear := cumsum(dayofyear), 
       by=list(year, location, model, emission)]
}



