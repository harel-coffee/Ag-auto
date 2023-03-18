col = "CDI"
names(A_85)[names(A_85) == col] <- 'val'
df <- merge(A_85[c('model', 'lat', 'long', 'location', 'val')], A_obs[c('location', col)], all.x=T)
df$val <- df$val - df[, col]


df_hiPosChange <- df %>% filter(val >= 0.5)
df_hiNegChange <- df %>% filter(val <= -0.1)

df_NoChange <- df %>% filter(val <= 0.08)
df_NoChange <- df_NoChange %>% filter(val >= -0.08)


df_mean <- aggregate(val ~ lat+long+location, data=df, mean)