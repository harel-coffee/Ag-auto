library(foreign)
                          
data_b <- read.dbf("/Users/hn/Documents/GitHub/large_4_GitHub/email_pruett_kaushik/not_min_VICID_CO.DBF", as.is=T)
setnames(data_b, old= colnames(data_b), new= tolower(colnames(data_b)))
setnames(data_b, old=c("vicclat", "vicclon"), new=c("lat", "long"))

data_b <- data_b %>% 
          filter(state %in% c("WA")) %>%
          data.table()

# we need the data that are in three counties: 
# Skagit County:057, Snohomish: 061, Whatcom:073
# 
int_fips <- c(53057, 53061, 53073)
data_b <- data_b %>% 
          filter(fips %in% int_fips) %>%
          data.table()

# plot to make sure:
states <- map_data("state")
WA_state <- subset(states, region %in% c("washington"))

data_b %>%
ggplot() +
geom_polygon(data = WA_state, 
             aes(x=long, y=lat, group = group),
                 fill = "grey", color = "black", size=0.5) +
geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

data_b$location <- paste0(data_b$lat, "_", data_b$long)
data_b <- subset(data_b, select=c(fips, location))

out_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
write.table(data_b, 
            file = paste0(out_dir, "three_counties.csv"),
            row.names = FALSE, na="", 
            col.names=TRUE, sep=",")

