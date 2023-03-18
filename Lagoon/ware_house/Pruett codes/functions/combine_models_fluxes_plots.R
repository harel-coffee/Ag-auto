library("tidyverse")
library("zoo")
library("lubridate")
library("cowplot")

source("functions/combine_models.R")
source("functions/combine_models_fluxes.R")
source("functions/wsu_colors.R")

# df <- combine_models("data_48.46875_-122.40625", climate_proj = "rcp85") # DRY
# df <- combine_models("data_48.71875_-121.09375", climate_proj = "rcp85") # WET

## Fluxes
df <- combine_models_fluxes("fluxes_48.46875_-122.40625", climate_proj = "A1B") %>% 
  mutate(combined = runoff + baseflow,
         group = factor(group, levels = c("hist", "2040s", "2060s", "2080s"))) # DRY
# df <- combine_models_fluxes("fluxes_48.71875_-121.09375", climate_proj = "A1B") %>% mutate(combined = runoff + baseflow) # WET

#### MONTH PROB ####

df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
  group_by(month, year) %>%
  summarise(combined = sum(combined)) %>% 
  group_by(month) %>%
  mutate(prob = rank(combined)/(n()+1)) %>%
  summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
            combined_90 = nth(combined, which.min(abs(prob-0.9))),
            combined_95 = nth(combined, which.min(abs(prob-0.95))))

df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
  filter(!is.na(group))

p_monthly_exceedance_val <- df_monthly_prob %>% 
  filter(group != "hist", month >= 10 | month <= 3) %>% 
  group_by(group, month, model, year) %>%
  summarise(combined = sum(combined),
            combined_80 = first(combined_80),
            combined_90 = first(combined_90),
            combined_95 = first(combined_95),
            time_stamp = first(time_stamp)) %>% 
  group_by(group, month, model) %>% 
  mutate(prob = rank(-combined)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(combined-combined_80))),
            prob_90 = nth(prob, which.min(abs(combined-combined_90))),
            prob_95 = nth(prob, which.min(abs(combined-combined_95))),
            month_name = factor(first(month(time_stamp, label = TRUE)), 
                                levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
  gather(exceedance, prob, -month, -model, -group, -month_name) %>% 
  group_by(group, month) %>% 
  mutate(prob_mean = mean(prob)) %>% 
  ggplot() +
  geom_point(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.50) +
  geom_point(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.75) +
  geom_point(aes(x = group, y = prob_95, color = group, fill = group), alpha = 1.00) +
  geom_boxplot(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.25) +
  geom_boxplot(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.50) +
  geom_boxplot(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.75) +
  geom_line(aes(x = group, y = prob_80_mean, group = model, color = "hist", fill = "hist"), alpha = 0.50) +
  geom_line(aes(x = group, y = prob_90_mean, group = model, color = "hist", fill = "hist"), alpha = 0.75) +
  geom_line(aes(x = group, y = prob_95_mean, group = model, color = "hist", fill = "hist"), alpha = 1.00) +
  geom_hline(aes(yintercept = 0.20), alpha = 0.50, color = "black") +
  geom_hline(aes(yintercept = 0.10), alpha = 0.75, color = "black") +
  geom_hline(aes(yintercept = 0.05), alpha = 1.00, color = "black") +
  facet_grid(.~month_name, switch = "x") +
  labs(x = " ", y = "Monthly Prob (Dry)") +
  theme_classic(base_size = size) +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.title = element_blank(), 
        legend.position = "none",
        strip.background = element_blank(), 
        strip.placement = "outside") +
  # scale_y_continuous(breaks = c(0.05, 0.1, 0.2), limits = c(0, 0.3)) +
  scale_color_wsu(palette = 'rev') +
  scale_fill_wsu(palette = 'rev')

df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
  group_by(group, month, model, year) %>% 
  summarise(combined = sum(combined)) %>% 
  mutate(prob = rank(combined)/(n()+1)) %>%
  summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
            combined_90 = nth(combined, which.min(abs(prob-0.9))),
            combined_95 = nth(combined, which.min(abs(prob-0.95))))

p_octmar_exceedance_val <- df %>%
  filter(month >= 10 | month <= 3, group != "hist") %>% 
  group_by(group, month, model, year) %>% 
  summarise(combined = sum(combined)) %>% 
  group_by(group, model) %>%
  mutate(prob = rank(-combined)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_80))),
            prob_90 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_90))),
            prob_95 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_95)))) %>%  
  group_by(group) %>% 
  mutate(prob_80_mean = mean(prob_80),
         prob_90_mean = mean(prob_90),
         prob_95_mean = mean(prob_95)) %>% 
  ggplot() +
  geom_point(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.50) +
  geom_point(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.75) +
  geom_point(aes(x = group, y = prob_95, color = group, fill = group), alpha = 1.00) +
  geom_boxplot(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.25) +
  geom_boxplot(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.50) +
  geom_boxplot(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.75) +
  geom_line(aes(x = group, y = prob_80_mean, group = model, color = "hist", fill = "hist"), alpha = 0.50) +
  geom_line(aes(x = group, y = prob_90_mean, group = model, color = "hist", fill = "hist"), alpha = 0.75) +
  geom_line(aes(x = group, y = prob_95_mean, group = model, color = "hist", fill = "hist"), alpha = 1.00) +
  geom_hline(aes(yintercept = 0.20), alpha = 0.50, color = "black") +
  geom_hline(aes(yintercept = 0.10), alpha = 0.75, color = "black") +
  geom_hline(aes(yintercept = 0.05), alpha = 1.00, color = "black") +
  labs(x = "Oct - Mar") +
  theme_classic(base_size = size) +
  theme(axis.text = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks = element_blank(),
        legend.title = element_blank(),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_blank(),
        axis.title.x = element_text(size = 12, face = "plain"),
        axis.line.y = element_blank()) +
  scale_color_wsu(palette = 'rev') +
  scale_fill_wsu(palette = 'rev') +
  # ylim(0,.3) +
  NULL

p_prob_monthly_dry <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(5.5, 1), axis = 'b')

p_all <- plot_grid(p_prob_monthly_wet, p_prob_monthly_dry, ncol = 1, align = "vh")

save_plot("figures/exceedance_probability_fluxes.pdf", p_all, base_aspect_ratio = 1.2, scale = 2, limitsize = FALSE)
