library(stringr)

combine_models_fluxes <- function(file_name, climate_proj){
  
  models <- c("ccsm3", "cgcm3.1_t47", "cnrm_cm3", "echam5", "echo_g", "hadcm", "historical")
  
  read_RDS <- function(model){
    
    if(model == "historical"){
      df <- readRDS(paste0("data/pruett/runoff/RDS/historical/", file_name, ".RDS"))
      df <- mutate(df, time_stamp = ymd(paste(year, month, "01", sep="-")),
                   water_year = year(time_stamp %m+% months(3)),
                   model = model, climate_proj = NA, group = "hist")
      return(df)
    } else {
      df_futr <- readRDS(paste0("data/pruett/runoff/RDS/", model, "_", climate_proj, "/", file_name, ".RDS")) %>% 
        filter(year >= 2016)
      
      df_NA <- df_futr %>% 
               filter(year <= 2025 || year >= 2095) %>%
               mutate(group = NA)
      
      df_2040 <- df_futr %>% filter(year >= 2025, year <= 2055) %>%
        mutate(group = "2040s")
      
      df_2060 <- df_futr %>% filter(year >= 2045, year <= 2075) %>%
        mutate(group = "2060s")
      
      df_2080 <- df_futr %>% filter(year >= 2065, year <= 2095) %>%
        mutate(group = "2080s")
      
      futr <- bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
        mutate(group = as.factor(group))
      
      df <- mutate(futr, time_stamp = ymd(paste(year, month, "01", sep="-")),
                   water_year = year(time_stamp %m+% months(3)),
                   model = model, climate_proj = climate_proj)
      
      return(df)
    }
    
  }
  map(models, read_RDS) %>% bind_rows() %>% mutate(model = as.factor(model))
}
