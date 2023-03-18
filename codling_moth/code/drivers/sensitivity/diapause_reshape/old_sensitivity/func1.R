a = 102.6077
b = 1.306483
x_shift = 16.95815

len = 101
steps = seq(from=0, to=1, by=(1 - 0)/len)

p0_multiplier = (1 - steps)^2
p1_multiplier = 2 * (1 - steps) * steps
p2_multiplier = steps^2

p0 = c(x_shift + log(log(a/100))/b, 100)
p1 = c(x_shift + log(log(a/100))/b, 10)
p2 = c(c + log(log(a/0.00001))/b, 0.00001)

new_x = p0[1] * p0_multiplier + p1[1] * p1_multiplier + p2[1] * p2_multiplier
new_y = p0[2] * p0_multiplier + p1[2] * p1_multiplier + p2[2] * p2_multiplier

margin_s = 4
par(mar=c(margin_s, margin_s, margin_s, margin_s))
plot(new_x, new_y, type="l", 
     xlab = "Day length", 
     xlim=c(14, 20),
     ylab="% diapause-induced",
     ylim=c(0, 100)
     )

