diapause<-data.table(seq(10, 20, .05))
diapause$PerDI = 102.6077 * exp(-exp(-(-1.306483) * (diapause$V1 - 16.95815)))
plot(x=diapause$V1, y = diapause$PerDI)

rm(list = ls())
library(ggplot2)

diapause_line_function_h <- function(x, prod_scale = 102.6077, expo_p1 = 1.306483, expo_p2=14){
	prod_scale * exp(-exp(expo_p1 * (x - expo_p2)))
}

diapause_line_function <- function(x, prod_scale = 102.6077, expo_p1 = 1.306483, expo_p2=16.95815){
	prod_scale * exp(-exp(expo_p1 * (x - expo_p2)))
}

y_min = 0
y_max = 105
x_min = -10
x_max = 25

ggplot(data.frame(x = c(x_min, x_max)), aes(x = x)) +
stat_function(fun = diapause_line_function) +
# stat_function(fun = diapause_line_function_h, color="red") +
theme_bw() + 
scale_y_continuous(limits = c(y_min, y_max), breaks=seq(y_min, y_max, by=10), minor_breaks = NULL) + 
theme(panel.grid.major = element_line(size=0.2),
	  panel.grid.minor = element_blank(),
      #legend.title = element_text(face="plain", size=12),
      #legend.text = element_text(size=10),
      #legend.position = "bottom",
      #strip.text = element_text(size=12, face="plain"),
      #axis.text = element_text(face="plain", size=10),
      #axis.title.x = element_text(face= "plain", size=16, margin = margin(t=10, r=0, b=0, l=0)),
      
    )


