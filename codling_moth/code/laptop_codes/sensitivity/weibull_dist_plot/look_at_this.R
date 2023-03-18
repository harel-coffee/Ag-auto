# http://t-redactyl.io/blog/2016/03/creating-plots-in-r-using-ggplot2-part-9-function-plots.html
ggplot(data.frame(x = c(0, 1)), aes(x = x)) +
        stat_function(fun = dnorm, args = list(0.2, 0.1),
                      aes(colour = "Group 1")) +
        stat_function(fun = dnorm, args = list(0.7, 0.05),
                      aes(colour = "Group 2")) +
        scale_x_continuous(name = "Probability",
                              breaks = seq(0, 1, 0.2),
                              limits=c(0, 1)) +
        scale_y_continuous(name = "Frequency") +
        ggtitle("Normal function curves of probabilities") +
        scale_colour_manual("Groups", values = c("deeppink", "dodgerblue3"))