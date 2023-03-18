library(dplyr)
library(data.table)


library(broom)
library(tidyverse)
# Create test data.
dat = data.frame(count=c(10, 60, 20, 50),
                 ring=c("A", "A", "B", "B"),
                 category=c("C", "D", "C", "D"))

# compute pvalue
# cs.pvalue <- dat %>%
#              spread(value = count,key=category) %>%
#              ungroup() %>%
#              select(-ring) %>%
#              chisq.test() %>%
#              tidy()

# cs.pvalue <- dat %>% 
#              spread(value = count,key=category) %>% 
#              select(-ring) %>%
#              fisher.test() %>% 
#              tidy() %>% 
#              full_join(cs.pvalue)

# compute fractions
dat %<>% group_by(ring) %>% mutate(fraction = count / sum(count),
                                   ymax = cumsum(fraction),
                                   ymin = c(0,ymax[1:length(ymax)-1]))
# Add x limits
baseNum <- 4
#numCat <- length(unique(dat$ring))
dat$xmax <- as.numeric(dat$ring) + baseNum
dat$xmin = dat$xmax - 1

# plot
p2 = ggplot(dat, aes(fill=category,
                     alpha = ring,
                     ymax=ymax, 
                     ymin=ymin, 
                     xmax=xmax, 
                     xmin=xmin)) +
    geom_rect(colour="grey30") +
    coord_polar(theta="y") +
    # geom_text(inherit.aes = F,
    #           x=c(-1,1),
    #           y=0,
    #           data = cs.pvalue, aes(label = paste(method,
    #                                               "\n",
    #                                               format(p.value,
    #                                                      scientific = T,
    #                                                      digits = 2))))+
  xlim(c(0, 6)) +
  theme_bw() +
  theme(panel.grid=element_blank()) +
  theme(axis.text=element_blank()) +
  theme(axis.ticks=element_blank(),
        panel.border = element_blank()) +
  labs(title="Customized ring plot") + 
  scale_fill_brewer(palette = "Set1") +
  scale_alpha_discrete(range = c(0.5,0.9))

p2


plot_fresh_pie <- function(fin_data){
  similarities <- unlist(seperate_1D_similarities(fin_data))
  press_sim <- similarities[1]
  precip_sim <- similarities[2]

  dat = data.frame(count = c(press_sim, (1-press_sim), 
                               precip_sim, (1-precip_sim)),
                   ring = c("pest pressure ring", "pest pressure ring", 
                            "precipitation ring", "precipitation ring"),
                   category = c("variable", "variable complement", 
                                "variable", "variable complement"))
  # compute fractions
  dat %<>% group_by(ring) %>% 
           mutate(fraction = count / sum(count),
                  ymax = cumsum(fraction),
                  ymin = c(0,ymax[1:length(ymax)-1]))
  # Add x limits
  baseNum <- 4
  # numCat <- length(unique(dat$ring))
  dat$xmax <- as.numeric(dat$ring) + baseNum
  dat$xmin = dat$xmax - 1

  p2 = ggplot(dat, aes(fill = category, alpha = ring,
                       ymax = ymax, ymin = ymin, 
                       xmax = xmax, xmin = xmin)) +
       geom_rect(colour = "grey30") +
       coord_polar(theta = "y") +
       xlim(c(0, 6)) +
       theme_bw() +
       theme(panel.grid = element_blank()) +
       theme(axis.text = element_blank()) +
       theme(axis.ticks = element_blank(),
             panel.border = element_blank()) +
       labs(title = "Customized ring plot") + 
       scale_fill_brewer(palette = "Set1") +
       scale_alpha_discrete(range = c(0.3, 0.8)) + 
       theme(plot.title = element_text(size=18, face="bold"),
             # because its ring, r, l, b are messed up 
             #            (do NOT count on the followings)
             #                   r: is actually left
             #                   l: is actually bottom
             #                   b: is actually right
             # plot.margin = unit(c(t=-10, b=-50, l=-50, r=10), "pt"), # almost good
             # plot.margin = unit(c(t=-10, b=-10, l=-30, r=0), "pt"), # better
             plot.margin = unit(c(t=-10, b=-10, l=-30, r=0), "pt"),
             panel.grid=element_blank(),
             legend.spacing.x = unit(.2, 'pt'),
             legend.title = element_blank(),
             legend.position = "none",
             legend.key.size = unit(1.6, "line"),
             legend.text = element_blank())
  return(p2)
}