

combine_models <- function(file_name, climate_proj){
  
  models <- list.dirs("data/pruett/RDS", recursive = FALSE, full.names = FALSE)
  
  read_RDS <- function(model){

    if(model == "historical"){
      df <- readRDS(paste0("data/pruett/RDS/historical/", file_name, ".rds"))
      df <- mutate(df, time_stamp = ymd(paste(year, month, day, sep="-")),
                   water_year = year(time_stamp %m+% months(3)),
                   model = model, climate_proj = NA)
      return(df)
    } else {
      df_futr <- readRDS(paste0("data/pruett/RDS/", model, "/", climate_proj, "/", file_name, ".rds")) %>% 
        filter(year >= 2016)
      
      df_NA <- df_futr %>% filter(year <= 2025 || year >= 2095) %>%
        mutate(group = NA)
      
      df_2040 <- df_futr %>% 
                 filter(year >= 2025, year <= 2055) %>%
                 mutate(group = "2040s")
      
      df_2060 <- df_futr %>% 
                 filter(year >= 2045, year <= 2075) %>%
                 mutate(group = "2060s")
      
      df_2080 <- df_futr %>% 
                 filter(year >= 2065, year <= 2095) %>%
                 mutate(group = "2080s")
      
      x <- nrow(df_2080)
      
      futr <- bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
        mutate(group = as.factor(group))

      df <- mutate(futr, time_stamp = ymd(paste(year, month, day, sep="-")),
                   water_year = year(time_stamp %m+% months(3)),
                   model = model, climate_proj = climate_proj)

      return(df)
    }

  }
  map(models, read_RDS) %>% 
    bind_rows() %>% 
    mutate(model = as.factor(model),
           group = factor(group, levels = c("hist", "2040s", "2060s", "2080s")))
}






