# list model directories
model_dirs <- list.dirs("/data/hydro/jennylabcommon2/metdata/VIC_ext_maca_v2_binary_westUSA", recursive = FALSE)
models <- list.dirs("/data/hydro/jennylabcommon2/metdata/VIC_ext_maca_v2_binary_westUSA", full.names = FALSE, recursive = FALSE)

historical_dir <- "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016"

climate_proj <- c("rcp45", "rcp85")

file_names <- readRDS("data/pruett/RDS/spatial.rds")
file_names <- file_names$file_name


# read Binary Data

### read binary files ###

# build file path
build_path <- function(model_dir, climate_proj, file_name){
  paste0(model_dir, climate_proj, file_name)
}


merge_timeframe <- function(model){
  
  if(model == "historical"){
    df <- read_binary(paste0("data/pruett/RDS/historical/", file_name, ".rds"))
    df <- mutate(df, time_stamp = ymd(paste(year, month, day, sep="-")),
                 water_year = year(time_stamp %m+% months(3)),
                 model = model, climate_proj = NA)
    return(df)
  } else {
    df_futr <- readRDS(paste0("data/pruett/RDS/", model, "/", climate_proj, "/", file_name, ".rds")) %>% 
      filter(year >= 2016)
    
    df_NA <- df_futr %>% filter(year <= 2025 || year >= 2095) %>%
      mutate(group = NA)
    
    df_2040 <- df_futr %>% filter(year >= 2025, year <= 2055) %>%
      mutate(group = "2040s")
    
    df_2060 <- df_futr %>% filter(year >= 2045, year <= 2075) %>%
      mutate(group = "2060s")
    
    df_2080 <- df_futr %>% filter(year >= 2065, year <= 2095) %>%
      mutate(group = "2080s")
    
    futr <- bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
      mutate(group = as.factor(group))
    
    df <- mutate(futr, time_stamp = ymd(paste(year, month, day, sep="-")),
                 water_year = year(time_stamp %m+% months(3)),
                 model = model, climate_proj = climate_proj)
    
    return(df)
  }
  
}