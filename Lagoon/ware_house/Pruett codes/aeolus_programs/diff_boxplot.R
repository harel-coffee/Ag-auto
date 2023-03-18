library(tidyverse)
library(lubridate)
library(stringr)
library(cowplot)
library(rgdal)
library(sf)

# read data ####
df <- read_rds("data/precip_summary.rds") 


med_yearly <- df %>% 
  group_by(lat, lng, climate_proj, group, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, climate_proj, group) %>% 
  summarise(precip = median(precip))

med_hist <- med_yearly %>% 
  ungroup() %>% 
  filter(group == "hist") %>% 
  select(lat, lng, hist_precip = precip)

p_yearly <- med_yearly %>% 
  filter(group != "hist") %>% 
  full_join(med_hist) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip) %>% 
  ggplot() +
  geom_boxplot(aes(x = climate_proj, y = precip, fill = group)) +
  scale_y_continuous(labels = scales::percent) +
  labs(y = "Percent Change", 
       title = "Difference from Historical Median Annual Precip")

med_octmar <- df %>% 
  filter(!between(month, 4, 9)) %>%
  mutate(date_time = ymd(paste(year, month, "01", sep = "-")),
         water_year = year(date_time %m+% months(3))) %>%
  group_by(lat, lng, climate_proj, group, model, water_year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, climate_proj, group) %>% 
  summarise(precip = median(precip))

med_hist_octmar <- med_octmar %>% 
  ungroup() %>% 
  filter(group == "hist") %>% 
  select(lat, lng, hist_precip = precip)

p_octmar <- med_octmar %>% 
  filter(group != "hist") %>% 
  full_join(med_hist_octmar) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip) %>% 
  ggplot() +
  geom_boxplot(aes(x = climate_proj, y = precip, fill = group)) +
  scale_y_continuous(labels = scales::percent) +
  labs(y = "Percent Change", 
       title = "Difference from Historical Median Oct- Mar Precip")

# seasonal boxplots ####
df_season <- df %>%
  mutate(season = case_when(month <= 3 ~ "Jan-Mar",
                            between(month, 4, 6) ~ "Apr-Jun",
                            between(month, 7, 9) ~ "Jul-Sep",
                            month >= 10 ~ "Oct-Dec")) %>% 
  group_by(lat, lng, year, model, climate_proj, group, season) %>%
  summarise(precip = sum(precip)) %>%
  group_by(lat, lng, climate_proj, group, season) %>%
  summarise(precip = median(precip))

df_season_hist <- df_season %>% filter(group == "hist") %>% 
  ungroup() %>% 
  select(lat, lng, season, hist_precip = precip) %>% 
  full_join(df_season) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip)

plot_seasonal_boxplot <- function(df, sel_season){
  df %>% filter(season == sel_season, group != "hist") %>% 
    ggplot() +
    geom_boxplot(aes(x = climate_proj, y = precip, fill = group)) +
    scale_y_continuous(labels = scales::percent) +
    labs(y = "Percent Change", 
         title = paste("Difference from Historical Median", sel_season,"Precip"))
}

# combine plots ####
df_plots <- tibble(data = list(df_season_hist),
                   seasons = as_factor(unique(df_season$season), "Jan-Mar")) %>% 
  mutate(plot = map2(data, seasons, plot_seasonal_boxplot))


p <- plot_grid(df_plots$plot[[1]], 
               df_plots$plot[[2]], 
               df_plots$plot[[3]], 
               df_plots$plot[[4]], 
               p_octmar, p_yearly, ncol = 1)


save_plot("figures/diff_hist_boxplot.pdf", p, base_aspect_ratio = 0.8, scale = 4)






