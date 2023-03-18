
param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
cluster_data <- read.csv(paste0(param_dir, 
                                "5_clust_with_elevation_info.csv"),
                         as.is=TRUE)

cluster_data$cluster <- factor(cluster_data$cluster, 
                               levels=c("Western coastal", 
                                        "Cascade foothills", 
                                        "Northwest Cascades", 
                                        "Northcentral Cascades", 
                                        "Northeast Cascades"))

ax_txt_size <- 10; ax_ttl_size <- 14; 
color_ord = c("royalblue3", "steelblue1", "maroon3", "red", "black")

the <- theme(plot.margin = unit(c(t=.1, r=0.1, b=.1, l=.1), "cm"),
             panel.border = element_rect(fill=NA, size=.3),
             panel.grid.major = element_line(size = 0.05),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.35, "line"), panel.spacing.y = unit(.5, 'line'),
             legend.title = element_blank(),
             legend.text = element_text(size = 8, face="bold"),
             legend.background=element_blank(),
             legend.position = c(.2, .92),
             plot.title = element_text(size=8, face = "bold",
                                       margin = margin(t=.15, r=.1, b=0, l=0, "line")),
             plot.subtitle = element_text(face = "bold"),
             axis.ticks = element_line(size = .1, color = "black"),
             axis.text.y = element_text(size = ax_txt_size, face = "bold", color = "black"),
             axis.text.x = element_text(size = ax_txt_size, face = "bold", color="black",
                                        margin=margin(t=.05, r=5, l=5, b=0,"pt")),
             axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                         margin = margin(t=0, r=8, b=0, l=0)),
             axis.title.x = element_text(size = ax_ttl_size, face = "bold", 
                                         margin = margin(t=8, r=0, b=0, l=0)))

cluster_plt <- ggplot(cluster_data, aes(x=elevation, y=ann_prec_mean, color=cluster)) +
               geom_point() + 
               ylab("annual precipitation mean (mm)") +
               the +
               guides(colour = guide_legend(override.aes = list(size=3))) +
               scale_color_manual(values = color_ord, name = "Precip.")

plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
ggsave(filename = "cluster_visualization_5.png", 
       plot = cluster_plt, device = "png",
       width = 5, height = 9, 
       units = "in", dpi=600,
       path = plot_dir)


