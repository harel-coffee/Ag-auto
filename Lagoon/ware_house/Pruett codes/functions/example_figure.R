library(tidyverse)
library(cowplot)
library(ggrepel)

df <- read_rds("data/example/data_47.78125_-121.78125.rds") %>% 
  filter(climate_proj == "rcp85")

ggplot(df) +
  geom_jitter(aes(x = group, y = prob, color = group, fill = group), width = .2) +
  geom_line(aes(x = group, y = prob_median, group = model, color = "hist")) +
  geom_text(aes(label = hist_prob, x = "2080s", y = hist_prob), nudge_x = unit(.5, "lines"), nudge_y = .5) +
  geom_hline(aes(yintercept = hist_prob)) +
  facet_grid(exceedance~., switch = "x", scales = "free_y") +
  labs(x = "Oct - Mar") +
  theme_minimal(base_size = size) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.y = element_blank(),
        legend.title = element_blank(),
        strip.background = element_blank(),
        axis.title.x = element_text(size = 12, face = "plain")) +
  scale_color_wsu(palette = 'rev') +
  scale_fill_wsu(palette = 'rev') +
  scale_y_continuous(labels=scaleFUN)

