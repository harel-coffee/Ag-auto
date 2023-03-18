rm(list=ls())
library(data.table)
library(dplyr)
options(digit=9)

write_dir_utah = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/utah_model_stats/"
write_dir_dynamic = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/dynamic_model_stats/"

model_names = c("utah", "dynamic")

for (model in model_names){
  if (model=="dynamic"){
    setwd(write_dir_dynamic)
    write_dir = write_dir_dynamic
  } else {
    setwd(write_dir_utah)
    write_dir = write_dir_utah
  }

  summary_comp <- data.table(readRDS(paste0(write_dir, "summary_comp.rds")))

  # newdata <- subset(mydata, sex=="m" & age > 25, select=weight:income)
  summary_comp$location = paste0(as.character(summary_comp$lat), as.character(summary_comp$long))

  summary_comp <- subset(summary_comp, select=c("sum_J1", "model", "scenario", 
                          "year", "location"))

  # This includes historical, rcp45 and rcp85
  modeled_medians <- summary_comp %>%
             filter(model != "observed") %>%
             group_by(location, year, model, scenario) %>%
             summarise_at(.funs = funs(med = median), vars(sum_J1)) %>% data.table()

  observed_medians <- summary_comp %>%
            filter(model == "observed") %>%
            group_by(location, year) %>%
            summarise_at(.funs = funs(med = median), vars(sum_J1)) %>% data.table()

  scenarios = c("historical", "rcp45", "rcp85")
  models = unique(modeled_medians$model)
  locations = unique(modeled_medians$location)

  n_rows = length(scenarios) * length(models) * length(locations)
  modeled_slopes = data.frame(matrix(ncol = 4, nrow = n_rows))
  colnames(modeled_slopes) <- c("location", "model", "scenario", "slope")

  modeled_slopes$location = locations
  modeled_slopes$model = models
  modeled_slopes$scenario = scenarios

  for (row in 1:dim(modeled_slopes)[1]){
    loc = modeled_slopes[row, "location"]
    mod = modeled_slopes[row, "model"]
    sce = modeled_slopes[row, "scenario"]

    curr_data = modeled_medians %>% filter(location== loc & 
                         model   == mod & 
                         scenario== sce)
    linearMod = lm(med ~ year, data=curr_data)
    modeled_slopes[row, "slope"] = summary(linearMod)$coefficients[2, 1]
    rm(curr_data)
  }
   
  write_dir = paste0(write_dir, "slopes/")
  write.table(modeled_slopes, file = paste0(write_dir, model, "_modeled_slopes.csv"),
        row.names=FALSE, 
        col.names=TRUE, 
        sep=",")

  saveRDS(modeled_slopes, paste0(write_dir, model, "_modeled_slopes.rds"))

  ######## observed

  observed_medians <- summary_comp %>%
            filter(model == "observed") %>%
            group_by(location, year) %>%
            summarise_at(.funs = funs(med = median), vars(sum_J1)) %>% data.table()

  locations = unique(observed_medians$location)

  n_rows = length(locations)
  observed_slopes = data.frame(matrix(ncol = 2, nrow = n_rows))
  colnames(observed_slopes) <- c("location", "slope")

  observed_slopes$location = locations

  for (row in 1:dim(observed_slopes)[1]){
    loc = observed_slopes[row, "location"]

    curr_data = observed_medians %>% filter(location== loc)
    linearMod = lm(med ~ year, data=curr_data)
    observed_slopes[row, "slope"] = summary(linearMod)$coefficients[2, 1]
    rm(curr_data)
  }

  write.table(observed_slopes, file = paste0(write_dir, model, "_observed_slopes.csv"),
        row.names=FALSE, 
        col.names=TRUE, 
        sep=",")

  saveRDS(observed_slopes, paste0(write_dir, model, "_observed_slopes.rds"))
}