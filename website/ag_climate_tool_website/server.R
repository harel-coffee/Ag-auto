# Create the server logic

shinyServer(function(input, output, session) {
  ## Interactive Map ## 
  
  map = leaflet(data = cbStates) %>%
        addPolygons(color="black", weight=3, stroke = TRUE)%>%
        addTiles(
               urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
                 ) 
  output$map <- renderLeaflet(map)
  
  pal = colorBin(palette = "Reds", domain = as.numeric(crb_locations$num_crops), bins = 8, pretty=FALSE)
  leafletProxy("map") %>%
    addLegend("topleft", pal = pal, values = NULL, title = "Number of Crops") 
  
  ## List of crops based on location ##
  #output$selected_crops <- select_crops()
  
  ###################################################################################################
  #Plots
  ###################################################################################################
  ################################################################################################### 
  
    output$ET_plotly <- renderPlotly({ 
    
    withProgress(message = 'Loading...', value = 0, {
    # Set the value based on number of loads that are happening
    n <- 8
    
    #1 pull data on click and store in temporary place 
    #Increment the loading progress
    incProgress(1/n)
    
    loc <- select_location()
    
    historical <- select_historical()
    future <- select_future() 
    
    #2 pull elevation  for that location 
    #Increment the loading progress
    incProgress(1/n)
    
    elev <-  elevlocs %>% filter(latitude == loc$latitude & longitude == loc$longitude) %>% select(Elevation)
    
    
    #3)add yday column for DOY variable in ET calculations 
    #Increment the loading progress
    incProgress(1/n)
    
    historical$yday <- yday(historical$date)
    future$yday <- yday(future$date)
    
    historic <- historical # %>% 
     # group_by(month, yday) %>% 
     # summarize(temp_max = mean(temp_max), 
     #           temp_min = mean(temp_min), 
     #           windspeed = mean(windspeed), 
     #           SPH = mean(SPH), 
     #           SRAD = mean(SRAD), 
     #           RMAX = mean(RMAX), 
     #           RMIN = mean(RMIN))
    
    future2040 <- filter(future, between(year, 2038,2042))# %>% 
     # group_by(month, yday) %>% 
     # summarize(temp_max = mean(temp_max), 
     #           temp_min = mean(temp_min), 
     #           windspeed = mean(windspeed), 
     #           SPH = mean(SPH), 
     #           SRAD = mean(SRAD), 
     #           RMAX = mean(RMAX), 
     #           RMIN = mean(RMIN))
    
    future2060 <- filter(future,  between(year, 2058,2062)) # %>% 
     # group_by(month, yday) %>% 
     # summarize(temp_max = mean(temp_max), 
     #           temp_min = mean(temp_min), 
     #           windspeed = mean(windspeed), 
     #           SPH = mean(SPH), 
     #           SRAD = mean(SRAD), 
     #           RMAX = mean(RMAX), 
     #           RMIN = mean(RMIN))
    
    future2080 <- filter(future,  between(year, 2078,2082)) # %>% 
     # group_by(month, yday) %>% 
     # summarize(temp_max = mean(temp_max), 
     #           temp_min = mean(temp_min), 
     #           windspeed = mean(windspeed), 
     #           SPH = mean(SPH), 
     #           SRAD = mean(SRAD), 
     #           RMAX = mean(RMAX), 
     #           RMIN = mean(RMIN))
    
    rm(historical)
    rm(future)
    
    #4) walk through each day (row?) and calculate ET and add to column
    #Increment the loading progress
    incProgress(1/n)
    
    hit <- addET(data.frame(historic), lat = loc$latitude, elevation = elev)
    
    fut2040 <- addET(data.frame(future2040), lat = loc$latitude, elevation = elev)
    fut2060 <- addET(data.frame(future2060), lat = loc$latitude, elevation = elev)
    fut2080 <- addET(data.frame(future2080), lat = loc$latitude, elevation = elev)
    
    #4.5 getting the sum data for ET

     hit <- data.frame(hit %>% group_by(year, month) %>% 
      mutate(sumET = sum(ET)) %>% 
      select(year, month, sumET) %>% 
      unique() 
    )
      
    
    fut2040 <- data.frame(fut2040 %>% group_by(year, month) %>% 
      mutate(sumET = sum(ET)) %>% 
      select(year, month, sumET) %>% 
      unique()
    )
    
    
    fut2060 <- data.frame(fut2060 %>% group_by(year, month) %>% 
      mutate(sumET = sum(ET)) %>% 
      select(year, month, sumET) %>% 
      unique()
    )
    
    
    fut2080 <- data.frame(fut2080 %>% group_by(year, month) %>% 
      mutate(sumET = sum(ET)) %>% 
      select(year, month, sumET) %>% 
      unique()
    )
    
    #5) add source indexing
    #Increment the loading progress
    incProgress(1/n)
    
    hit$source <- "historic"
    fut2040$source <- "2040"
    fut2060$source <- "2060"
    fut2080$source <- "2080"
    
    
    #6) bind together so we can group by
    #Increment the loading progress
    incProgress(1/n)
    
    ETdat <- rbind(hit, fut2040, fut2060, fut2080)
    
    #7)grouping data
    #Increment the loading progress
    incProgress(1/n)
    
    d <- ETdat 

    #8) chart will change with future vs historical mean daily ET.
    #Increment the loading progress
    incProgress(1/n)
    
    #making specific color pallet
    pal <- c(palette3$hist, palette3$fut2040, palette3$fut2060, palette3$fut2080)
    pal <- setNames(pal, c("historic", "2040", "2060", "2080"))
    
    plot.ET <- plot_ly(data =d, y = sumET, x = month, color = source, colors = pal, type = "box") %>% layout(boxmode = "group")
    
    #plot.ET <- plot_ly( data = d, y = ~meanET, x = ~month, group_by = source, color = ~factor(source), colors = pal, style = "line") %>% 
     # layout(title = "Historical vs Future Evapotransporation", 
      #       xaxis = list(title = "Days of the Year"), 
       #      yaxis = list(title = "Evapotransporation (units)")
        #     ) 
    
    #Increment the loading progress
    incProgress(1/n)
    
    })
    
    plot.ET 
  
  })



#  output$mahalanobis_map <- renderImage({
#    loc <- select_location()
#    image_src <- paste0(paste(paste("data/mahalanobis", 
#                                    future_model, 
#                                    future_rcp, 
#                                    "combined",
#                                    "data", sep="/"),
#                              loc$latitude, loc$longitude, sep="_"), ".png")
#    
#    list(src = image_src,
#         contentType = 'image/png',
#         width = 900,
#         height = 900,
#         alt = "image of climate similarity.")
#  }, deleteFile = FALSE)
  
#put in place of above matching function
   output$match_county_map <- renderImage({
     loc <- select_location()
  
  
     mod <- input$mod
     rcp <- input$rcp
     year <- input$matchyear
  
     
     fips <- county %>% filter(latitude == loc$latitude & longitude == loc$longitude) %>% select(fips)
     #espect map image files stored in directory of this structure: "2040", "2060", or "2080
       ## data/matching/CNRM-CM5/rcp45/2040/fips.png
     image_match <- paste0(paste("data/matches/west_trimmed", 
                                  mod, 
                                  rcp, 
                                  fips, sep="/"
                                ),
                          ".png"  )
  
     list(src = image_match,
          contentType = 'image/png',
          width = "100%",
          # height = 600,
          alt = "Most similar county to future projection.")
    }, deleteFile = FALSE)

#Put in place when table file is present 
   output$matches_table <- renderTable({
     
     loc <- select_location() 
     mod <- input$mod
     rcp <- input$rcp
     year <- input$matchyear
    
     
     clicked_fips <- as.character(county %>% filter(latitude == loc$latitude & longitude == loc$longitude) %>% select(fips))
     clicked_name <- county %>% filter(fips == clicked_fips) %>% select(county, state) %>% unique()
     
     clicked_comatch <- matchdat %>% 
       filter(f_fips == clicked_fips, model == mod, rcp == input$rcp, year == as.numeric(input$matchyear) & region != "fr")
     
     # matched_name <- county %>% 
     #   filter(fips == clicked_comatch$h_fips) %>% 
     #   select(county, state) 
     
     matched_name <- county %>% 
       rename(h_fips = fips) %>% 
       right_join(clicked_comatch) %>% 
       arrange(desc(h_ps)) %>% 
       select(county, state) 
     
     matched_name <- clicked_name %>% 
       bind_rows(matched_name) %>% 
       unique() %>%
       mutate(name = paste0(county, ", ", state)) %>%
       select(name)
         
     tabledat <- data.frame(matched_name)
     row.names(tabledat) <- c("Selected", "First Best", "Second Best", "Third Best")
     names(tabledat) <- c("County")
     tabledat
   })
      
   output$match_county_table <- renderTable({
     
     loc <- select_location() 
     mod <- input$mod
     rcp <- input$rcp
     year <- input$matchyear
    
     
     clicked_fips <- as.character(county %>% filter(latitude == loc$latitude & longitude == loc$longitude) %>% select(fips))
     clicked_name <- county %>% filter(fips == clicked_fips) %>% select(county, state) %>% unique()
         
     clicked_comatch <- matchdat %>% filter(f_fips == clicked_fips, model == mod, rcp == input$rcp, year == as.numeric(input$matchyear) & region != "fr") 
 
     #pulling out slected county for selected column 
     #we only want one row, so use unique
     clicked_co <-  clicked_comatch %>% 
          select(contains("f_")) %>% 
          unique() %>%
          mutate(f_county = clicked_name$county, f_state = clicked_name$state) %>%
          mutate_if(is.numeric, round, digits = 2)
     names(clicked_co) <- substring(names(clicked_co), 3) 
     
     #using matchdat to pull county name and state for best match county 

     matched_name <- county %>% filter(fips == clicked_comatch$h_fips) %>% select(county, state) %>% unique()
    
     matched_co <-  clicked_comatch %>% 
       select(contains("h_")) %>% 
       mutate(h_county = matched_name$county, h_state = matched_name$state) %>% 
       mutate_if(is.numeric, round, digits = 2)
     names(matched_co) <- substring(names(matched_co), 3)
     
     # Only use the best match
     paired <- rbind(clicked_co, matched_co[1,]) 
     paired <- data.frame(t(paired))
     names(paired) <- c("Selected", "Best Match")

     removerows <- c("fips") 
     tabledat <- paired[-which(rownames(paired) %in% removerows),] 
     row.names(tabledat) <- match_varnames

     tabledat
     
     })




  # output$gdd_forecast_plotly <- renderPlotly({
  #   #dependent on when "gddButton" is pressed
  #   
  #   withProgress(message = 'Loading...', value = 0, {
  #     # Set the value based on number of loads that are happening
  #     n <- 7
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     #UI data isolated so that we don't replot until gddButton is clicked.
  #     planting_date <- isolate(input$planting_date)
  #     
  #     crop = subset_crop()
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     #Data
  #     hd <- summarize_historical_over_day()
  #     hy <- summarize_historical_over_year()
  #     pd <- summarize_present_over_day()
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     #Stage Dates
  #     stage_dates <- hd[, c("monthday", "first_emergence_day", "first_maturity_day"), with=FALSE]
  #     stage_dates[, date := as.Date(paste0(this_year, "-", monthday))]
  #     
  #     #Calculate GEFS and CFS GDD
  #     cgdd_yesterday <- pd[date==(today-1), cgdd]
  #     gefs <- select_gefs()
  #     gefs[, gdd := calc_gdd(temp, 
  #                            temp_base=crop$temp.base, 
  #                            temp_max=crop$temp.max, 
  #                            temp_min=crop$temp.min), by=list(ens, date)]
  #     gefs[date < planting_date, gdd := 0]
  #     gefs[, cgdd := cumsum(gdd) + cgdd_yesterday, by=list(ens, year(date))]
  #     gefs_final <- gefs[, .(cgdd_mean = mean(cgdd, na.rm=T),
  #                            cgdd_min = min(cgdd, na.rm=T),
  #                            cgdd_max = max(cgdd, na.rm=T)), 
  #                        by=list(date)]
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     #NMME data
  #     nmme1 <- select_nmme()
  #     nmme2 <- merge(nmme1, hy[, c("monthday", "temp_mean"), with=FALSE], by="monthday")
  #     nmme2[, temp := temp + temp_mean]
  #     nmme2[, gdd := calc_gdd(temp, 
  #                             temp_base=crop$temp.base, 
  #                             temp_max=crop$temp.max, 
  #                             temp_min=crop$temp.min), by=list(ens, date)]
  #     nmme2[date < planting_date, gdd := 0]
  #     nmme2[!is.na(gdd) & year(date) == year(today), cgdd := cgdd_yesterday + cumsum(gdd), by=list(ens, year(date))]
  #     nmme_final <- nmme2[, .(cgdd_mean = mean(cgdd, na.rm=T),
  #                             cgdd_min = min(cgdd, na.rm=T),
  #                             cgdd_max = max(cgdd, na.rm=T)
  #     ), by=list(date)]
  #     setkey(nmme_final, date)
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     hist_last_frost_pct = temp_hist_frost()[[1]]
  #     hist_first_frost_pct = temp_hist_frost()[[2]]
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #     #Plots for the GDD/Precip Page
  #     gddly <- plot_gdd_forecast(crop, 
  #                                hy[year(date) == this_year], 
  #                                pd[year(date) == this_year], 
  #                                gefs_final[year(date) == this_year], 
  #                                nmme_final[year(date) == this_year],
  #                                stage_dates,
  #                                hist_last_frost_pct = hist_last_frost_pct,
  #                                hist_first_frost_pct = hist_first_frost_pct,
  #                                colors = color_scheme,
  #                                engine="plotly")
  #     
  #     #Increment the loading progress
  #     incProgress(1/n)
  #     
  #   })
  #   
  #   gddly
  # })
  
  output$storm_outlook_plotly <- renderPlotly({
    precip <- summarize_historical_over_day()
    
    precip <- precip[, max_event := (max(precipitation)), by=year]
    precip$max_event / 24
    precip <- calc_summary(precip, var =  max_event, group_var = year)
    
    five = (-1) * (sqrt(6) / pi) * (0.5772 + log(log(5 /( 5 - 1))))
    ten = (-1) * (sqrt(6) / pi) * (0.5772 + log(log(10 /( 10 - 1))))
    fifteen = (-1) * (sqrt(6) / pi) * (0.5772 + log(log(15 /( 15 - 1))))
    twenty = (-1) * (sqrt(6) / pi) * (0.5772 + log(log(5 /( 5 - 1))))
    twentyfive = (-1) * (sqrt(6) / pi) * (0.5772 + log(log(5 /( 5 - 1))))
    
    
    xtFive = precip$max_event_mean + (five*precip$max_event_sd)
    xtTen = precip$max_event_mean + (ten*precip$max_event_sd)
    xtFifteen = precip$max_event_mean + (fifteen*precip$max_event_sd)
    xtTwenty = precip$max_event_mean + (twenty*precip$max_event_sd)
    xtTwentyfive = precip$max_event_mean + (twentyfive*precip$max_event_sd)
    
  })
  
  output$gdd_outlook_plotly <- renderPlotly({
    
    withProgress(message = 'Loading...', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 10
      
      #Increment the loading progress
      incProgress(1/n)
      
      #UI data isolated so that we don't replot until gddButton is clicked.
      #planting_date <- isolate(input$planting_date_longterm)
      
      crop <- subset_crop()
      #Data
      hd <- summarize_historical_over_day()
      if(leap_year(this_year)) hd <- hd[monthday != "02-29"]
      hy <- summarize_historical_over_year()
      
      fd <- summarize_future_over_day()
      if(leap_year(this_year)) fd <- fd[monthday != "02-29"]
      
      #Increment the loading progress
      incProgress(1/n)
      
      fy2040 <- summarizeOverYear(fd[year > 2025 & year <= 2055]) #collapse across years so 365 records
      fy2040$date <- as.Date(paste0(this_year, "-", fy2040$monthday), format = "%Y-%m-%d")
      setkey(fy2040, date)
      
      fy2060 <- summarizeOverYear(fd[year > 2045 & year <= 2075]) #collapse across years so 365 records
      fy2060$date <- as.Date(paste0(this_year, "-", fy2060$monthday), format = "%Y-%m-%d")
      setkey(fy2060, date)
      
      fy2080 <- summarizeOverYear(fd[year > 2065 & year <= 2095]) #collapse across years so 365 records
      fy2080$date <- as.Date(paste0(this_year, "-", fy2080$monthday), format = "%Y-%m-%d")
      setkey(fy2080, date)
      
      
      #Stage Dates
      stage_dates <- rbind(
        hd[, c("year", "monthday", "first_emergence_day", "first_maturity_day"), with=FALSE],
        fd[, c("year", "monthday", "first_emergence_day", "first_maturity_day"), with=FALSE]
      )
      stage_dates[, date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")]
      setkey(stage_dates, date)
      
      #Increment the loading progress
      incProgress(1/n)
      
      hist_last_frost_pct = temp_hist_frost()[[1]]
      hist_first_frost_pct = temp_hist_frost()[[2]]
      fut_last_frost_pct2040 = temp_fut_frost()[[1]]
      fut_first_frost_pct2040 = temp_fut_frost()[[4]]
      fut_last_frost_pct2060 = temp_fut_frost()[[2]]
      fut_first_frost_pct2060 = temp_fut_frost()[[5]]
      fut_last_frost_pct2080 = temp_fut_frost()[[3]]
      fut_first_frost_pct2080 = temp_fut_frost()[[6]]
      
      #Increment the loading progress
      incProgress(1/n)
      
      d <- data.frame("last_frost" = temp_hist_frost()[[1]]$last_frost, "first_frost" = temp_hist_frost()[[2]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      hist_med_len <- median(d$length)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[1]]$last_frost, "first_frost" = temp_fut_frost()[[4]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2040_med_len <- median(d$length)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[2]]$last_frost, "first_frost" = temp_fut_frost()[[5]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2060_med_len <- median(d$length)
      
      #Increment the loading progress
      incProgress(1/n)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[3]]$last_frost, "first_frost" = temp_fut_frost()[[6]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2080_med_len <- median(d$length)
      
      #Increment the loading progress
      incProgress(1/n)
      
      #planting_date2 <- isolate(input$planting_date_longterm2)
      hd2 <- summarize_historical_over_day2()
      if(leap_year(this_year)) hd2 <- hd2[monthday != "02-29"]
      hy2 <- summarize_historical_over_year2()
      fd2 <- summarize_future_over_day2()
      if(leap_year(this_year)) fd2 <- fd2[monthday != "02-29"]
      
      #Increment the loading progress
      incProgress(1/n)
      
      fy20402 <- summarizeOverYear(fd2[year > 2025 & year <= 2055]) #collapse across years so 365 records
      fy20402$date <- as.Date(paste0(this_year, "-", fy20402$monthday), format = "%Y-%m-%d")
      setkey(fy20402, date)
      
      fy20602 <- summarizeOverYear(fd2[year > 2045 & year <= 2075]) #collapse across years so 365 records
      fy20602$date <- as.Date(paste0(this_year, "-", fy20602$monthday), format = "%Y-%m-%d")
      setkey(fy20602, date)
      
      fy20802 <- summarizeOverYear(fd2[year > 2065 & year <= 2095]) #collapse across years so 365 records
      fy20802$date <- as.Date(paste0(this_year, "-", fy20802$monthday), format = "%Y-%m-%d")
      setkey(fy20802, date)
      
      #Increment the loading progress
      incProgress(1/n)
      
      #Stage Dates
      stage_dates2 <- rbind(
        hd2[, c("year", "monthday", "first_emergence_day", "first_maturity_day"), with=FALSE],
        fd2[, c("year", "monthday", "first_emergence_day", "first_maturity_day"), with=FALSE]
      )
      stage_dates2[, date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")]
      setkey(stage_dates2, date)
      
      #Increment the loading progress
      incProgress(1/n)
      
      #Plots for the GDD/Precip Page
      gddly <- plot_gdd_outlook(crop, 
                                hy[year(date) == this_year], 
                                fy2040[year(date) == this_year], 
                                fy2060[year(date) == this_year], 
                                fy2080[year(date) == this_year], 
                                stage_dates = stage_dates,
                                
                                hy2[year(date) == this_year],
                                fy20402[year(date) == this_year],
                                fy20602[year(date) == this_year],
                                fy20802[year(date) == this_year],
                                stage_dates2 = stage_dates2,
                                
                                hist_last_frost_pct = hist_last_frost_pct, 
                                hist_first_frost_pct = hist_first_frost_pct,
                                fut_last_frost_pct2040 = fut_last_frost_pct2040, 
                                fut_first_frost_pct2040 = fut_first_frost_pct2040,
                                fut_last_frost_pct2060 = fut_last_frost_pct2060,
                                fut_first_frost_pct2060 = fut_first_frost_pct2060,
                                fut_last_frost_pct2080 = fut_last_frost_pct2080,
                                fut_first_frost_pct2080 = fut_first_frost_pct2080,
                                hist_med_len = hist_med_len, 
                                fut2040_med_len = fut2040_med_len,
                                fut2060_med_len = fut2060_med_len, 
                                fut2080_med_len = fut2080_med_len,
                                
                                colors = color_scheme,
                                engine="plotly")
      
      #Increment the loading progress
      incProgress(1/n)
      
    })
    
    gddly
  })
  
  
  
  
  
  
  output$gdd_similarity_plotly <- renderPlotly({
    withProgress(message = 'Loading...0/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 7
      
      #Isolated UI inputs
      crop <- isolate(subset_crop())
      nass <- subset_nass_yields()
      
      incProgress(1/n)
      
      ## Historical data
      hd <- summarize_historical_over_day()
      hy <- summarize_historical_over_year() 
      hm <- hd[, .(cgdd = sum(gdd)), by=list(month(date), year(date))]
      
      incProgress(1/n)
      
      ## Present Data plus NMME 
      pd <- summarize_present_over_day()
      
      incProgress(1/n)
      
      nmme1 <- select_nmme()
      nmme2 <- merge(nmme1, hy[, c("monthday", "temp_mean"), with=FALSE], by="monthday")
      nmme2[, temp := temp + temp_mean]
      nmme2[, gdd := calc_gdd(temp), by=list(date, ens)]
      nmme_final <- nmme2[date >= today, .(gdd = mean(gdd, na.rm=T)), 
                          by=list(date)]
      setkey(nmme_final, date)
      
      incProgress(1/n)
      
      #add the NMME and present
      cols <- c("date", "gdd")
      pres <- rbind(pd[, cols, with=FALSE], nmme_final[, cols, with=FALSE])
      
      pm <- pres[, .(cgdd = sum(gdd)),
                 by=list(month(date), year(date))][year == this_year]
      
      ## Combine and calculate distance
      dat <- calc_time_dissimilarity(hm, pm, "cgdd", this_year = this_year)
      
      cgdd_yld <- na.omit(merge(dat, nass, all.x = TRUE)) #[year<2017]
      cgdd_yld[, `:=`(dis_rank = rank(dis), maha_rank = rank(maha))]
      
      incProgress(1/n)
      
      p_plotly <- plot_time_dissimilarity(cgdd_yld, gdd_colors, 
                                          title_str = "GDD", legend_str = "Dissimilarity",
                                          xlab_str="", ylab_str="Yield", detrend = FALSE,
                                          engine = "plotly")
      incProgress(1/n)
    })
    
    p_plotly
  })
  
  output$precip_similarity_plotly <- renderPlotly({
    withProgress(message = 'Loading...1/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 6
      
      input$gddButton
      input$gddButton2
      #Isolated UI inputs
      crop <- isolate(subset_crop())
      nass <- subset_nass_yields()
      
      incProgress(1/n)
      
      ## Historical data
      hd <- summarize_historical_over_day()
      hy <- summarize_historical_over_year() 
      hm <- hd[, .(cprecip = sum(precipitation)), by=list(month(date), year(date))]
      
      incProgress(1/n)
      
      ## Present Data plus NMME 
      pd <- summarize_present_over_day()
      
      incProgress(1/n)
      
      nmme1 <- select_nmme()
      nmme2 <- merge(nmme1, hy[, c("monthday", "precip_mean"), with=FALSE], by="monthday")
      nmme2[, precip := precip + precip_mean]
      nmme_final <- nmme2[date >= today, .(precipitation = mean(precip, na.rm=T)), 
                          by=list(date)]
      setkey(nmme_final, date)
      
      incProgress(1/n)
      
      #add the NMME and present
      cols <- c("date", "precipitation")
      pres <- rbind(pd[, cols, with=FALSE], nmme_final[, cols, with=FALSE])
      
      pm <- pres[, .(cprecip = sum(precipitation)),
                 by=list(month(date), year(date))][year == this_year]
      
      ## Combine and calculate distance
      dat <- calc_time_dissimilarity(hm, pm, "cprecip", this_year = this_year)
      
      incProgress(1/n)
      
      cprecip_yld <- na.omit(merge(dat, nass, all.x = TRUE)) #[year<2017]
      cprecip_yld[, `:=`(dis_rank = rank(dis), maha_rank = rank(maha))]
      
      p_plotly <- plot_time_dissimilarity(cprecip_yld, precip_colors, 
                                          title_str = "Precipitation", legend_str = "Dissimilarity",
                                          xlab_str="", ylab_str="Yield", detrend = FALSE,
                                          engine = "plotly")
      
      incProgress(1/n)
    })
    
    p_plotly
  })
  
  output$gdd_similarity_detrended_plotly <- renderPlotly({
    withProgress(message = 'Loading...2/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 7
      
      input$gddButton
      input$gddButton2
      #Isolated UI inputs
      crop <- isolate(subset_crop())
      nass <- subset_nass_yields()
      
      incProgress(1/n)
      
      ## Historical data
      hd <- summarize_historical_over_day()
      hy <- summarize_historical_over_year() 
      hm <- hd[, .(cgdd = sum(gdd)), by=list(month(date), year(date))]
      
      incProgress(1/n)
      
      ## Present Data plus NMME 
      pd <- summarize_present_over_day()
      
      incProgress(1/n)
      
      nmme1 <- select_nmme()
      nmme2 <- merge(nmme1, hy[, c("monthday", "temp_mean"), with=FALSE], by="monthday")
      nmme2[, temp := temp + temp_mean]
      nmme2[, gdd := calc_gdd(temp), by=list(date, ens)]
      nmme_final <- nmme2[date >= today, .(gdd = mean(gdd, na.rm=T)), 
                          by=list(date)]
      setkey(nmme_final, date)
      
      incProgress(1/n)
      
      #add the NMME and present
      cols <- c("date", "gdd")
      pres <- rbind(pd[, cols, with=FALSE], nmme_final[, cols, with=FALSE])
      
      pm <- pres[, .(cgdd = sum(gdd)),
                 by=list(month(date), year(date))][year == this_year]
      
      incProgress(1/n)
      
      ## Combine and calculate distance
      dat <- calc_time_dissimilarity(hm, pm, "cgdd", this_year = this_year)
      
      cgdd_yld <- na.omit(merge(dat, nass, all.x = TRUE)) #[year<2017]
      cgdd_yld[, `:=`(dis_rank = rank(dis), maha_rank = rank(maha))]
      
      incProgress(1/n)
      
      p_plotly <- plot_time_dissimilarity(cgdd_yld, gdd_colors, 
                                          title_str = "GDD",
                                          xlab_str="", ylab_str="Yield (Difference from Trend)", 
                                          engine = "plotly")
      
      incProgress(1/n)
    })
    
    p_plotly
  })
  
  output$precip_similarity_detrended_plotly <- renderPlotly({
    withProgress(message = 'Loading...3/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 6
      
      input$gddButton
      input$gddButton2
      #Isolated UI inputs
      crop <- isolate(subset_crop())
      nass <- subset_nass_yields()
      
      incProgress(1/n)
      
      ## Historical data
      hd <- summarize_historical_over_day()
      hy <- summarize_historical_over_year() 
      hm <- hd[, .(cprecip = sum(precipitation)), by=list(month(date), year(date))]
      
      incProgress(1/n)
      
      ## Present Data plus NMME 
      pd <- summarize_present_over_day()
      
      incProgress(1/n)
      
      nmme1 <- select_nmme()
      nmme2 <- merge(nmme1, hy[, c("monthday", "precip_mean"), with=FALSE], by="monthday")
      nmme2[, precip := precip + precip_mean]
      nmme_final <- nmme2[date >= today, .(precipitation = mean(precip, na.rm=T)), 
                          by=list(date)]
      setkey(nmme_final, date)
      
      incProgress(1/n)
      
      #add the NMME and present
      cols <- c("date", "precipitation")
      pres <- rbind(pd[, cols, with=FALSE], nmme_final[, cols, with=FALSE])
      
      incProgress(1/n)
      
      pm <- pres[, .(cprecip = sum(precipitation)),
                 by=list(month(date), year(date))][year == this_year]
      
      ## Combine and calculate distance
      dat <- calc_time_dissimilarity(hm, pm, "cprecip", this_year = this_year)
      
      cprecip_yld <- na.omit(merge(dat, nass, all.x = TRUE)) #[year<2017]
      cprecip_yld[, `:=`(dis_rank = rank(dis), maha_rank = rank(maha))]
      
      incProgress(1/n)
      
      p_plotly <- plot_time_dissimilarity(cprecip_yld, precip_colors, 
                                          title_str = "Precipitation",
                                          xlab_str="", ylab_str="Yield (Difference from Trend)", 
                                          engine = "plotly")
      
      incProgress(1/n)
    })
    
    p_plotly
  })
  
  ########### SPATIAL DISTANCE PLOT ###########
  output$spatialDissimilarityPlot <- renderPlot({
    
    #Progress bar
    #     withProgress(message = 'Creating Analogue Maps...',
    #                  detail = 'Please hold...', value = 0, {
    #                    for (i in 1:15) {
    #                      incProgress(1/15)
    #                      Sys.sleep(0.25)
    #                    }
    #                  })
    #     
    loc <- select_location()
    
    #create data.table
    fut <- spatial_future()
    cnames <- c(names(fut), "lat", "lon")
    fut <- rbind(colMeans(fut[year >= (2035 - 10) & year <= (2035 + 10)]),
                 colMeans(fut[year >= (2085 - 10) & year <= (2085 + 10)]))
    fut <- data.frame(cbind(fut, loc$latitude, loc$longitude))
    names(fut) <- cnames
    # spatial_historical is a static dataset loaded in global.R
    dat <- data.table(rbind(fut, spatial_historical[, cnames], fill=TRUE))
    #make 4 plots
    plts <- list()
    plts[]
    years <- c(2085, 2035)
    for(j in 1:length(years)) {
      y <- years[j]
      y2 <- years[which(years != y)]
      d2 <- dat[year!=y]
      mdat <- d2[, -c("year", "lat", "lon"), with=FALSE]
      nrows <- nrow(mdat)
      
      #center and covariance
      center <- unlist(mdat[1,])
      sx_inv <- pseudoinverse(cov(mdat))
      
      res <- data.table(data.frame(
        lat = d2$lat,
        lon = d2$lon,
        dis = dissimilarity(mdat, center, collapse=FALSE),
        maha = mahalanobis(mdat, center, sx_inv, inverted=TRUE)
      ))
      res[, `:=`(dis_rank = rank(dis), maha_rank = rank(maha))]
      setkey(res, lat, lon)
      
      d_maha <- res[maha_rank <= (as.integer(input$analogue_number) + 1)][order(maha_rank)]
      d_dis  <- res[dis_rank <= (as.integer(input$analogue_number) + 1)][order(dis_rank)]
      
      plts[[paste0("maha_", j)]] <- plot_spatial_distance(d = d_maha, title = paste("Mahalanobis", y2))
      plts[[paste0("dis_", j)]]  <- plot_spatial_distance(d = d_dis, title = paste("Dissimilarity", y2))
    }
    p <- grid.arrange(plts$maha_1, plts$maha_2, plts$dis_1, plts$dis_2,
                      top = textGrob("GDD Climate Analogues", gp = gpar(fontsize=20,font=1)),
                      ncol = 2, nrow = 2)
    return(p)
  })
  
  
  
  
  
  
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  ####################################################################################################################################
  
  output$precip_forecast_snow <- renderPlotly({
    
    withProgress(message = 'Loading...0/3', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 4
      
      #Increment the loading progress
      incProgress(1/n)
      
      hist = precip_hist() 
      snow = precip_snow() 
      rain = precip_rain() 
      gefs = precip_gefs() 
      nmme = precip_nmme()
      
      #Increment the loading progress
      incProgress(1/n)
      
      ht = max(c(max(hist$snow), max(hist$rain), 
                 max(snow$cum_snow), max(rain$cum_rain), 
                 max(gefs$max_snow), max(gefs$max_rain), 
                 max(nmme$max_snow), max(nmme$max_rain)))
      
      #Increment the loading progress
      incProgress(1/n)
      
      p <- plot_precip_forecast_snow(hist = hist, snow = snow, gefs = gefs, nmme = nmme, ht = ht)
      
      #Increment the loading progress
      incProgress(1/n)
      
      p
    })
  })
  
  output$precip_forecast_rain <- renderPlotly({
    
    withProgress(message = 'Loading...1/3', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 4
      
      #Increment the loading progress
      incProgress(1/n)
      
      hist = precip_hist() 
      snow = precip_snow() 
      rain = precip_rain() 
      gefs = precip_gefs() 
      nmme = precip_nmme()
      
      incProgress(1/n)
      
      ht = max(c(max(hist$snow), max(hist$rain), 
                 max(snow$cum_snow), max(rain$cum_rain), 
                 max(gefs$max_snow), max(gefs$max_rain), 
                 max(nmme$max_snow), max(nmme$max_rain)))
      
      incProgress(1/n)
      
      p <- plot_precip_forecast_rain(hist = hist, rain = rain, gefs = gefs, nmme = nmme, ht = ht)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$precip_forecast_total <- renderPlotly({
    withProgress(message = 'Loading...2/3', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 3
      
      #Increment the loading progress
      incProgress(1/n)
      
      pd <- data.frame("date" = precip_rain()$date, 
                       "cum_rain" = precip_rain()$cum_rain, 
                       "cum_snow" = precip_snow()$cum_snow)
      
      incProgress(1/n)
      
      p <- plot_precip_forecast_total(hist = precip_hist(),
                                      pd = pd,
                                      gefs = precip_gefs(),
                                      nmme = precip_nmme())
      incProgress(1/n)
    })
    
    p
  })
  
  output$precip_outlook_snow <- renderPlotly({
    withProgress(message = 'Loading...0/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 3
      
      hist = precip_hist()
      fut3_40 = precip_fut()[[1]]
      fut3_60 = precip_fut()[[2]]
      fut3_80 = precip_fut()[[3]]
      
      incProgress(1/n)
      
      ht = max(c(max(hist$snow), max(hist$rain),
                 max(fut3_40$snow), max(fut3_40$rain),
                 max(fut3_60$snow), max(fut3_60$rain),
                 max(fut3_80$snow), max(fut3_80$rain)))
      
      incProgress(1/n)
      
      p <- plot_precip_outlook_snow(hist = hist, 
                                    fut3_40 = fut3_40,
                                    fut3_60 = fut3_60,
                                    fut3_80 = fut3_80,
                                    ht = ht)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$precip_outlook_rain <- renderPlotly({
    withProgress(message = 'Loading...1/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 3
      
      hist = precip_hist()
      fut3_40 = precip_fut()[[1]]
      fut3_60 = precip_fut()[[2]]
      fut3_80 = precip_fut()[[3]]
      
      incProgress(1/n)
      
      ht = max(c(max(hist$snow), max(hist$rain),
                 max(fut3_40$snow), max(fut3_40$rain),
                 max(fut3_60$snow), max(fut3_60$rain),
                 max(fut3_80$snow), max(fut3_80$rain)))
      
      incProgress(1/n)
      
      p <- plot_precip_outlook_rain(hist = hist, 
                                    fut3_40 = fut3_40,
                                    fut3_60 = fut3_60,
                                    fut3_80 = fut3_80,
                                    ht = ht)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$precip_outlook_total <- renderPlotly({
    withProgress(message = 'Loading...2/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 2
      
      hist = precip_hist()
      fut3_40 = precip_fut()[[1]]
      fut3_60 = precip_fut()[[2]]
      fut3_80 = precip_fut()[[3]]
      
      incProgress(1/n)
      
      p <- plot_precip_outlook_total(hist = precip_hist(),  
                                     fut3_40 = fut3_40,
                                     fut3_60 = fut3_60,
                                     fut3_80 = fut3_80)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$precip_outlook_frequency <- renderPlotly({
    withProgress(message = 'Loading...3/4', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 2
      
      incProgress(1/n)
      Sys.sleep(0.1)
      
      p <- plot_precip_outlook_frequency(fut3_40 = precip_fut3()[[1]],
                                         fut3_60 = precip_fut3()[[2]],
                                         fut3_80 = precip_fut3()[[3]],
                                         hist3 = precip_hist3())
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$temp_forecast <- renderPlotly({
    
    d <- data.frame("last_frost" = temp_hist_frost()[[1]]$last_frost, "first_frost" = temp_hist_frost()[[2]]$first_frost)
    d$length <- d$first_frost - d$last_frost
    hist_med_len <- median(d$length)
    
    hist = temp_hist()
    pd = temp_pd()
    gefs = temp_gefs()
    nmme = temp_nmme()
    hist_last_frost_pct = temp_hist_frost()[[1]]
    hist_first_frost_pct = temp_hist_frost()[[2]]
    
    #left is the parameter name, right is the passed in parameter
    plot_temp_forecast(
      hist = hist,
      pd = pd,
      gefs = gefs,
      nmme = nmme,
      hist_last_frost_pct = hist_last_frost_pct,
      hist_first_frost_pct = hist_first_frost_pct,
      hist_med_len = hist_med_len)
  })
  
  #add_trace(data = d, x = monthday, y = temperature, mode="markers")
  output$temp_outlook1 <- renderPlotly({
    withProgress(message = 'Loading...0/5', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 4
      
      hist = summarize_historical_over_month()
      fut = summarize_future_over_month()
      
      incProgress(1/n)
      
      fut2040 <- fut[2040 - 15 < year & year <= 2040 + 15]
      fut2060 <- fut[2060 - 15 < year & year <= 2060 + 15]
      fut2080 <- fut[2080 - 15 < year & year <= 2080 + 15]
      
      incProgress(1/n)
      
      hist$date = as.Date(paste0(this_year, paste0("-", paste0(hist$month, "-15"))), format = "%Y-%m-%d")
      fut2040$date = as.Date(paste0(this_year, paste0("-", paste0(fut2040$month, "-15"))), format = "%Y-%m-%d")
      fut2060$date = as.Date(paste0(this_year, paste0("-", paste0(fut2060$month, "-15"))), format = "%Y-%m-%d")
      fut2080$date = as.Date(paste0(this_year, paste0("-", paste0(fut2080$month, "-15"))), format = "%Y-%m-%d")
      
      incProgress(1/n)
      
      p <- plot_temp_outlook1(hist = hist,
                              fut2040 = fut2040,
                              fut2060 = fut2060,
                              fut2080 = fut2080)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$temp_outlook2 <- renderPlotly({
    withProgress(message = 'Loading...1/5', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 4
      
      hist = summarize_historical_over_month()
      fut = summarize_future_over_month()
      
      incProgress(1/n)
      
      fut2040 <- fut[2040 - 15 < year & year <= 2040 + 15]
      fut2060 <- fut[2060 - 15 < year & year <= 2060 + 15]
      fut2080 <- fut[2080 - 15 < year & year <= 2080 + 15]
      
      incProgress(1/n)
      
      hist$date = as.Date(paste0(this_year, paste0("-", paste0(hist$month, "-15"))), format = "%Y-%m-%d")
      fut2040$date = as.Date(paste0(this_year, paste0("-", paste0(fut2040$month, "-15"))), format = "%Y-%m-%d")
      fut2060$date = as.Date(paste0(this_year, paste0("-", paste0(fut2060$month, "-15"))), format = "%Y-%m-%d")
      fut2080$date = as.Date(paste0(this_year, paste0("-", paste0(fut2080$month, "-15"))), format = "%Y-%m-%d")
      
      incProgress(1/n)
      
      p <- plot_temp_outlook2(hist = hist,
                              fut2040 = fut2040,
                              fut2060 = fut2060,
                              fut2080 = fut2080)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$temp_outlook3 <- renderPlotly({
    withProgress(message = 'Loading...2/5', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 8
      
      d <- data.frame("last_frost" = temp_hist_frost()[[1]]$last_frost, "first_frost" = temp_hist_frost()[[2]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      hist_med_len <- median(d$length)
      
      incProgress(1/n)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[1]]$last_frost, "first_frost" = temp_fut_frost()[[4]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2040_med_len <- median(d$length)
      
      incProgress(1/n)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[2]]$last_frost, "first_frost" = temp_fut_frost()[[5]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2060_med_len <- median(d$length)
      
      incProgress(1/n)
      
      d <- data.frame("last_frost" = temp_fut_frost()[[3]]$last_frost, "first_frost" = temp_fut_frost()[[6]]$first_frost)
      d$length <- d$first_frost - d$last_frost
      fut2080_med_len <- median(d$length)
      
      incProgress(1/n)
      
      hist = summarize_historical_over_month()
      fut = summarize_future_over_month()
      
      incProgress(1/n)
      
      fut2040 <- fut[2040 - 15 < year & year <= 2040 + 15]
      fut2060 <- fut[2060 - 15 < year & year <= 2060 + 15]
      fut2080 <- fut[2080 - 15 < year & year <= 2080 + 15]
      
      incProgress(1/n)
      
      hist$date = as.Date(paste0(this_year, paste0("-", paste0(hist$month, "-15"))), format = "%Y-%m-%d")
      fut2040$date = as.Date(paste0(this_year, paste0("-", paste0(fut2040$month, "-15"))), format = "%Y-%m-%d")
      fut2060$date = as.Date(paste0(this_year, paste0("-", paste0(fut2060$month, "-15"))), format = "%Y-%m-%d")
      fut2080$date = as.Date(paste0(this_year, paste0("-", paste0(fut2080$month, "-15"))), format = "%Y-%m-%d")
      
      incProgress(1/n)
      
      p <- plot_temp_outlook3(hist = hist,
                              fut2040 = fut2040,
                              fut2060 = fut2060,
                              fut2080 = fut2080)
      
      incProgress(1/n)
    })
    
    p
  })
  
  
  
  output$temp_heat_risk <- renderPlotly({
    withProgress(message = 'Loading...3/5', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 2
      
      hist = temp_hist_heat_risk()
      fut3_40 = temp_fut_heat_risk()[[1]]
      fut3_60 = temp_fut_heat_risk()[[2]]
      fut3_80 = temp_fut_heat_risk()[[3]]
      
      incProgress(1/n)
      
      p <- plot_temp_heat_risk(hist = hist, 
                               fut3_40 = fut3_40, 
                               fut3_60 = fut3_60, 
                               fut3_80 = fut3_80)
      
      incProgress(1/n)
    })
    
    p
  })
  
  output$temp_frost_risk <- renderPlotly({
    withProgress(message = 'Loading...4/5', value = 0, {
      # Set the value based on number of loads that are happening
      n <- 2
      
      hist = temp_hist_frost_risk()
      fut3_40 = temp_fut_frost_risk()[[1]]
      fut3_60 = temp_fut_frost_risk()[[2]]
      fut3_80 = temp_fut_frost_risk()[[3]]
      
      incProgress(1/n)
      
      p <- plot_temp_frost_risk(hist = hist, 
                                fut3_40 = fut3_40, 
                                fut3_60 = fut3_60, 
                                fut3_80 = fut3_80)
      
      incProgress(1/n)
    })
    
    p
  })
  
  ########### FROST HEAT DAYS PLOT ###########
  output$fhPlot <- renderPlot({
    plotFrostHeatDays(select_Historical(), input$temp.max)
  })
  
  ########### COUNTY PLOT ###########
  output$countyPlot <- renderRbokeh({
    #p <- plot_county(summarized_county())
    #return(p)
  })
  
  ########### COUNTY SUMMARY TABLE ###########
  output$countyTable <- renderDataTable({ summarized_county() })
  output$countyText <- renderPrint({ nass }) 
  
  #### KIRTI suggestion  #####3
  #get merged data of form
  # data<-grid year model(hist plus each future) modelaggregated(hist or future) futuregroup (2040s 2060s 2080s) allcolumns
  # melt
  #datamelted<-melt(data, id = c(year, modelaggregated, futuregroup))
  
  #datasubsetpercentages<-
  #datasubsetdayoftheyear<-
  
  ########## CODLING PLOT  ########## 
  
  output$adultCodling <- renderPlot({
    data_sets <- select_codling()
    
    if (length(data_sets) > 4)
    {
      withProgress(message = 'Loading...', value = 0, {
        n <- 12
        
        a <- plot_codling_moth.ggplot(data_sets, 27)
        incProgress(1/n)
        b <- plot_codling_moth.ggplot(data_sets, 28)
        incProgress(1/n)
        c <- plot_codling_moth.ggplot(data_sets, 29)
        incProgress(1/n)
        d <- plot_codling_moth.ggplot(data_sets, 30)
        incProgress(1/n)
        e <- plot_codling_moth.ggplot(data_sets, 31)
        incProgress(1/n)
        f <- plot_codling_moth.ggplot(data_sets, 32)
        incProgress(1/n)
        g <- plot_codling_moth.ggplot(data_sets, 33)
        incProgress(1/n)
        h <- plot_codling_moth.ggplot(data_sets, 34)
        incProgress(1/n)
        i <- plot_codling_moth.ggplot(data_sets, 35)
        incProgress(1/n)
        j <- plot_codling_moth.ggplot(data_sets, 36)
        incProgress(1/n)
        k <- plot_codling_moth.ggplot(data_sets, 37)
        incProgress(1/n)
        
        q <- grid.arrange(a, b, c, d, e, f, g, h, i, j, k,
                          layout_matrix = rbind(c(1, 1, 1, 1, 1),
                                                c(2, 2, 2, 2, 2),
                                                c(3, 3, 3, 3, 3),
                                                c(4, 4, 4, 4, 4),
                                                c(5, 5, 5, 5, 5),
                                                c(6, 6, 6, 6, 6),
                                                c(7, 7, 7, 7, 7),
                                                c(8, 8, 8, 8, 8),
                                                c(9, 9, 9, 9, 9),
                                                c(10, 10, 10, 10, 10),
                                                c(11, 11, 11, 11, 11)
                          ))
        incProgress(1/n)
      })
    }
    else
    {
      q <- NULL
    }
    
    q
  })
  
  output$adultCodling2 <- renderPlot({
    data_sets <- select_codling()
    
    if (length(data_sets) > 4)
    {
      withProgress(message = 'Loading...', value = 0, {
        n <- 12
        
        a <- plot_codling_moth.ggplot(data_sets, 2)
        incProgress(1/n)
        b <- plot_codling_moth.ggplot(data_sets, 3)
        incProgress(1/n)
        c <- plot_codling_moth.ggplot(data_sets, 4)
        incProgress(1/n)
        d <- plot_codling_moth.ggplot(data_sets, 5)
        incProgress(1/n)
        e <- plot_codling_moth.ggplot(data_sets, 6)
        incProgress(1/n)
        f <- plot_codling_moth.ggplot(data_sets, 7)
        incProgress(1/n)
        g <- plot_codling_moth.ggplot(data_sets, 8)
        incProgress(1/n)
        h <- plot_codling_moth.ggplot(data_sets, 9)
        incProgress(1/n)
        i <- plot_codling_moth.ggplot(data_sets, 10)
        incProgress(1/n)
        j <- plot_codling_moth.ggplot(data_sets, 11)
        incProgress(1/n)
        k <- plot_codling_moth.ggplot(data_sets, 12)
        incProgress(1/n)
        
        q <- grid.arrange(a, b, c, d, e, f, g, h, i, j, k,
                          layout_matrix = rbind(c(1, 1, 1, 1, 1),
                                                c(2, 2, 2, 2, 2),
                                                c(3, 3, 3, 3, 3),
                                                c(4, 4, 4, 4, 4),
                                                c(5, 5, 5, 5, 5),
                                                c(6, 6, 6, 6, 6),
                                                c(7, 7, 7, 7, 7),
                                                c(8, 8, 8, 8, 8),
                                                c(9, 9, 9, 9, 9),
                                                c(10, 10, 10, 10, 10),
                                                c(11, 11, 11, 11, 11)
                          ))
        incProgress(1/n)
      })
    }
    else
    {
      q <- NULL
    }
    
    q
  })
  
  output$larvaCodling <- renderPlot({
    data_sets <- select_codling()
    
    if (length(data_sets) > 4)
    {
      withProgress(message = 'Loading...', value = 0, {
        n <- 12
        
        a <- plot_codling_moth.ggplot(data_sets, 38)
        incProgress(1/n)
        b <- plot_codling_moth.ggplot(data_sets, 39)
        incProgress(1/n)
        c <- plot_codling_moth.ggplot(data_sets, 40)
        incProgress(1/n)
        d <- plot_codling_moth.ggplot(data_sets, 41)
        incProgress(1/n)
        e <- plot_codling_moth.ggplot(data_sets, 42)
        incProgress(1/n)
        f <- plot_codling_moth.ggplot(data_sets, 43)
        incProgress(1/n)
        g <- plot_codling_moth.ggplot(data_sets, 44)
        incProgress(1/n)
        h <- plot_codling_moth.ggplot(data_sets, 45)
        incProgress(1/n)
        i <- plot_codling_moth.ggplot(data_sets, 46)
        incProgress(1/n)
        j <- plot_codling_moth.ggplot(data_sets, 47)
        incProgress(1/n)
        k <- plot_codling_moth.ggplot(data_sets, 48)
        incProgress(1/n)
        
        q <- grid.arrange(a, b, c, d, e, f, g, h, i, j, k,
                          layout_matrix = rbind(c(1, 1, 1, 1, 1),
                                                c(2, 2, 2, 2, 2),
                                                c(3, 3, 3, 3, 3),
                                                c(4, 4, 4, 4, 4),
                                                c(5, 5, 5, 5, 5),
                                                c(6, 6, 6, 6, 6),
                                                c(7, 7, 7, 7, 7),
                                                c(8, 8, 8, 8, 8),
                                                c(9, 9, 9, 9, 9),
                                                c(10, 10, 10, 10, 10),
                                                c(11, 11, 11, 11, 11)
                          ))
        incProgress(1/n)
      })
    }
    else
    {
      q <- NULL
    }
    q
  })
  
  output$larvaCodling2 <- renderPlot({
    data_sets <- select_codling()
    
    if (length(data_sets) > 4)
    {
      withProgress(message = 'Loading...', value = 0, {
        n <- 12
        
        a <- plot_codling_moth.ggplot(data_sets, 13)
        incProgress(1/n)
        b <- plot_codling_moth.ggplot(data_sets, 14)
        incProgress(1/n)
        c <- plot_codling_moth.ggplot(data_sets, 15)
        incProgress(1/n)
        d <- plot_codling_moth.ggplot(data_sets, 16)
        incProgress(1/n)
        e <- plot_codling_moth.ggplot(data_sets, 17)
        incProgress(1/n)
        f <- plot_codling_moth.ggplot(data_sets, 18)
        incProgress(1/n)
        g <- plot_codling_moth.ggplot(data_sets, 19)
        incProgress(1/n)
        h <- plot_codling_moth.ggplot(data_sets, 20)
        incProgress(1/n)
        i <- plot_codling_moth.ggplot(data_sets, 21)
        incProgress(1/n)
        j <- plot_codling_moth.ggplot(data_sets, 22)
        incProgress(1/n)
        k <- plot_codling_moth.ggplot(data_sets, 23)
        incProgress(1/n)
        
        q <- grid.arrange(a, b, c, d, e, f, g, h, i, j, k,
                          layout_matrix = rbind(c(1, 1, 1, 1, 1),
                                                c(2, 2, 2, 2, 2),
                                                c(3, 3, 3, 3, 3),
                                                c(4, 4, 4, 4, 4),
                                                c(5, 5, 5, 5, 5),
                                                c(6, 6, 6, 6, 6),
                                                c(7, 7, 7, 7, 7),
                                                c(8, 8, 8, 8, 8),
                                                c(9, 9, 9, 9, 9),
                                                c(10, 10, 10, 10, 10),
                                                c(11, 11, 11, 11, 11)
                          ))
        incProgress(1/n)
      })
    }
    else
    {
      q <- NULL
    }
    q
  })
  
  output$otherCodling <- renderPlot({
    data_sets <- select_codling()
    
    if (length(data_sets) > 4)
    {
      withProgress(message = 'Loading...', value = 0, {
        n <- 4
        a <- plot_codling_moth.ggplot(data_sets, 26)
        incProgress(1/n)
        b <- plot_codling_moth.ggplot(data_sets, 49)
        incProgress(1/n)
        c <- plot_codling_moth.ggplot(data_sets, 50)
        incProgress(1/n)
        
        q <- grid.arrange(a, b, c,
                          layout_matrix = rbind(c(1, 1, 1, 1, 1),
                                                c(2, 2, 2, 2, 2),
                                                c(3, 3, 3, 3, 3)
                          ))
        incProgress(1/n)
      })
    }
    else
    {
      q <- NULL
    }
    q
  })
  
  
  output$averageCodlingGen1 <- renderImage({
    image_src <- "data/codling/images/Gen1.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingGen2 <- renderImage({
    image_src <- "data/codling/images/Gen2.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingGen3 <- renderImage({
    image_src <- "data/codling/images/Gen3.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingGen4 <- renderImage({
    image_src <- "data/codling/images/Gen4.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingInter1 <- renderImage({
    image_src <- "data/codling/images/InterAnnualVariability_Gen1.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingInter2 <- renderImage({
    image_src <- "data/codling/images/InterAnnualVariability_Gen2.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingInter3 <- renderImage({
    image_src <- "data/codling/images/InterAnnualVariability_Gen3.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingInter4 <- renderImage({
    image_src <- "data/codling/images/InterAnnualVariability_Gen4.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingSpatial1 <- renderImage({
    image_src <- "data/codling/images/SpatialVariability_Gen1.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingSpatial2 <- renderImage({
    image_src <- "data/codling/images/SpatialVariability_Gen2.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingSpatial3 <- renderImage({
    image_src <- "data/codling/images/SpatialVariability_Gen3.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  output$averageCodlingSpatial4 <- renderImage({
    image_src <- "data/codling/images/SpatialVariability_Gen4.jpg"
    
    list(src = image_src,
         contentType = 'image/jpg',
         width = 550,
         height = 550)
  }, deleteFile = FALSE)
  
  
  ###################################################################################################
  #Reactives
  ###################################################################################################
  
  precip_hist <- reactive({
    hist <- summarize_historical_over_day()
    hist$rain <- hist$precipitation
    hist$snow <- hist$precipitation
    hist[temperature <= 0.6]$rain = 0
    hist[0.6 < temperature &  temperature < 3.6]$rain = hist[0.6 < temperature &  temperature < 3.6]$rain * ((hist[0.6 < temperature &  temperature < 3.6]$temperature / 3) - 0.2)
    hist[3.6 <= temperature]$snow = 0
    hist[0.6 < temperature &  temperature < 3.6]$snow = hist[0.6 < temperature &  temperature < 3.6]$snow * (1 - ((hist[0.6 < temperature &  temperature < 3.6]$temperature / 3) - 0.2))
    hist <- hist[,  .(monthday = monthday,
                      date = date,
                      cum_snow = cumsum(snow), 
                      cum_rain = cumsum(rain)), by=list(year(date), month(date))]
    hist <- hist[,  .(monthday = monthday,
                      snow = max(cum_snow), 
                      rain = max(cum_rain)), by=list(year(date), month(date))]
    
    hist$date <- as.Date(paste0(this_year, paste0("-", hist$monthday)), format = "%Y-%m-%d")
    hist <- rbind(hist[monthday == "01-15"], hist[monthday == "02-15"], hist[monthday == "03-15"], hist[monthday == "04-15"], hist[monthday == "05-15"],
                  hist[monthday == "06-15"], hist[monthday == "07-15"], hist[monthday == "08-15"], hist[monthday == "09-15"], hist[monthday == "10-15"], 
                  hist[monthday == "11-15"], hist[monthday == "12-15"])
    hist <- hist[order(as.Date(hist$date, format="%Y-%m-%d")),]
    hist
  })
  
  precip_hist3 <- reactive({
    precip_risk = input$change_risk_range*25.4
    hist3 <- summarize_historical_over_day()
    hist3 <- hist3[order(as.Date(hist3$date, format="%Y-%m-%d")),]
    hist3 <- hist3[monthday != "02-29"]
    
    hist3$risk_day = 0
    hist3[precip_risk < precipitation]$risk_day = 1
    hist3[precip_risk >=  precipitation]$risk_day = 0
    hist3 <- hist3[,  .(monthday = monthday,
                        year = year,
                        month = month,
                        date = date,
                        risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    hist3 <- hist3[,  .(monthday = monthday,
                        year = year,
                        month = month,
                        risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    hist3$date <- as.Date(paste0(this_year, paste0("-", hist3$monthday)), format = "%Y-%m-%d")
    
    hist3 <- rbind(hist3[monthday == "01-15"], hist3[monthday == "02-15"], hist3[monthday == "03-15"], hist3[monthday == "04-15"], hist3[monthday == "05-15"],
                   hist3[monthday == "06-15"], hist3[monthday == "07-15"], hist3[monthday == "08-15"], hist3[monthday == "09-15"], hist3[monthday == "10-15"], 
                   hist3[monthday == "11-15"], hist3[monthday == "12-15"])
    hist3
  })
  
  precip_fut <- reactive({
    fut <- summarize_future_over_day()
    fut$rain <- fut$precipitation
    fut$snow <- fut$precipitation
    fut[temperature <= 0.6]$rain = 0
    fut[0.6 < temperature &  temperature < 3.6]$rain = fut[0.6 < temperature &  temperature < 3.6]$rain * ((fut[0.6 < temperature &  temperature < 3.6]$temperature / 3) - 0.2)
    fut[3.6 <= temperature]$snow = 0
    fut[0.6 < temperature &  temperature < 3.6]$snow = fut[0.6 < temperature &  temperature < 3.6]$snow * (1 - ((fut[0.6 < temperature &  temperature < 3.6]$temperature / 3) - 0.2))
    fut <- fut[,  .(monthday = monthday,
                    date = date,
                    cum_snow = cumsum(snow), 
                    cum_rain = cumsum(rain)), by=list(year(date), month(date))]
    fut <- fut[,  .(monthday = monthday,
                    snow = max(cum_snow), 
                    rain = max(cum_rain)), by=list(year(date), month(date))]
    
    fut$date <- as.Date(paste0(this_year, paste0("-", fut$monthday)), format = "%Y-%m-%d")
    fut <- rbind(fut[monthday == "01-15"], fut[monthday == "02-15"], fut[monthday == "03-15"], fut[monthday == "04-15"], fut[monthday == "05-15"],
                 fut[monthday == "06-15"], fut[monthday == "07-15"], fut[monthday == "08-15"], fut[monthday == "09-15"], fut[monthday == "10-15"], 
                 fut[monthday == "11-15"], fut[monthday == "12-15"])
    fut <- fut[order(as.Date(fut$date, format="%Y-%m-%d")),]
    
    fut3_40 <- fut[2040 - 15 < year & year <= 2040 + 15]
    fut3_60 <- fut[2060 - 15 < year & year <= 2060 + 15]
    fut3_80 <- fut[2080 - 15 < year & year <= 2080 + 15]
    
    list(fut3_40, fut3_60, fut3_80)
  })
  
  precip_fut3 <- reactive({
    precip_risk = input$change_risk_range*25.4
    fut3 <- summarize_future_over_day()
    fut3 <- fut3[order(as.Date(fut3$date, format="%Y-%m-%d")),]
    fut3 <- fut3[monthday != "02-29"]
    
    fut3$risk_day = 0
    fut3[precip_risk < precipitation]$risk_day = 1
    fut3[precip_risk >=  precipitation]$risk_day = 0
    fut3 <- fut3[,  .(monthday = monthday,
                      year = year,
                      month = month,
                      date = date,
                      risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    fut3 <- fut3[,  .(monthday = monthday,
                      year = year,
                      month = month,
                      risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    fut3$date <- as.Date(paste0(this_year, paste0("-", fut3$monthday)), format = "%Y-%m-%d")
    
    fut3 <- rbind(fut3[monthday == "01-15"], fut3[monthday == "02-15"], fut3[monthday == "03-15"], fut3[monthday == "04-15"], fut3[monthday == "05-15"],
                  fut3[monthday == "06-15"], fut3[monthday == "07-15"], fut3[monthday == "08-15"], fut3[monthday == "09-15"], fut3[monthday == "10-15"], 
                  fut3[monthday == "11-15"], fut3[monthday == "12-15"])
    
    fut3_40 <- fut3[2040 - 15 < year & year <= 2040 + 15]
    fut3_60 <- fut3[2060 - 15 < year & year <= 2060 + 15]
    fut3_80 <- fut3[2080 - 15 < year & year <= 2080 + 15]
    
    list(fut3_40, fut3_60, fut3_80)
  })
  
  precip_snow <- reactive({
    snow <- summarize_present_over_day()
    snow[3.6 <= temperature]$precipitation = 0
    snow[0.6 < temperature & temperature < 3.6]$precipitation = snow[0.6 < temperature & temperature < 3.6]$precipitation * (1 - ((snow[0.6 < temperature & temperature < 3.6]$temperature / 3) - .2))
    snow <- data.frame("date" = snow$date, "precip" = snow$precipitation)
    snow <- data.table(snow)
    snow <- snow[,  .(cum_snow = cumsum(precip), date = date), by=month(date)]
    snow
  })
  
  precip_rain <- reactive({
    rain <- summarize_present_over_day()
    rain[temperature <= 0.6]$precipitation = 0
    rain[0.6 < temperature & temperature < 3.6]$precipitation = rain[0.6 < temperature & temperature < 3.6]$precipitation * ((rain[0.6 < temperature & temperature < 3.6]$temperature / 3) - .2)
    rain <- data.frame("date" = rain$date, "precip" = rain$precipitation)
    rain <- data.table(rain)
    rain <- rain[,  .(cum_rain = cumsum(precip), date = date), by=month(date)]
    rain
  })
  
  precip_gefs <- reactive({
    gefs <- select_gefs()
    gefs <- gefs[is.na(precip) != TRUE]
    
    gefs$rain <- gefs$precip
    gefs$snow <- gefs$precip
    
    gefs[temp <= 0.6]$rain = 0
    gefs[0.6 < temp &  temp < 3.6]$rain = gefs[0.6 < temp &  temp < 3.6]$rain * ((gefs[0.6 < temp &  temp < 3.6]$temp / 3) - 0.2)
    gefs[3.6 <= temp]$snow = 0
    gefs[0.6 < temp &  temp < 3.6]$snow = gefs[0.6 < temp &  temp < 3.6]$snow * (1 - ((gefs[0.6 < temp &  temp < 3.6]$temp / 3) - 0.2))
    
    gefs <- gefs[, .(rain_mean = mean(rain), snow_mean = mean(snow)), by=list(date)]
    gefs <- gefs[,  .(date = date, cum_snow = cumsum(snow_mean), cum_rain = cumsum(rain_mean)), by=list(year(date), month(date))]
    gefs <- gefs[,  .(date = date, max_snow = max(cum_snow), max_rain = max(cum_rain)), by=list(year(date), month(date))]
    gefs <- gefs[order(as.Date(gefs$date, format="%Y-%m-%d")),]
    
    gefs
  })
  
  precip_nmme <- reactive({
    hy <- summarize_historical_over_year()
    nmme <- select_nmme()
    nmme[, `:=`(year = as.integer(format(date, "%y")))]
    nmme <- subset(nmme, year == format(Sys.Date(), "%y"))
    nmme2 <- merge(nmme, hy[, c("monthday", "precip_mean", "temp_mean"), with=FALSE], by="monthday")
    nmme2[, `:=`(precip = precip + precip_mean), by=list(date, ens)]
    nmme2[, `:=`(temp = temp + temp_mean), by=list(date, ens)]
    nmme <- nmme2[, .(temp = mean(temp, na.rm = TRUE)), by=monthday]
    nmme$precip <- nmme2[, .(precip = mean(precip, na.rm = TRUE)), by=monthday]$precip
    nmme2$rain <- nmme2$precip
    nmme2$snow <- nmme2$precip
    nmme2[temp <= 0.6]$rain = 0
    nmme2[0.6 < temp &  temp < 3.6]$rain = nmme2[0.6 < temp &  temp < 3.6]$rain * ((nmme2[0.6 < temp &  temp < 3.6]$temp / 3) - 0.2)
    nmme2[3.6 <= temp]$snow = 0
    nmme2[0.6 < temp &  temp < 3.6]$snow = nmme2[0.6 < temp &  temp < 3.6]$snow * (1 - ((nmme2[0.6 < temp &  temp < 3.6]$temp / 3) - 0.2))
    nmme <- nmme2[, .(rain_mean = mean(rain, na.rm=T), snow_mean = mean(snow, na.rm=T)), by=list(date)]
    nmme <- nmme[,  .(date = date, cum_snow = cumsum(snow_mean), cum_rain = cumsum(rain_mean)), by=list(year(date), month(date))]
    nmme <- nmme[,  .(date = date, max_snow = max(cum_snow), max_rain = max(cum_rain)), by=list(year(date), month(date))]
    nmme <- nmme[order(as.Date(nmme$date, format="%Y-%m-%d")),]
    nmme
  })
  
  temp_hist <- reactive({
    
    hd <- summarize_historical_over_day()
    hd$monthyear = paste0(hd$month, paste0("-", hd$year))
    hd$day <- as.numeric(strftime(hd$date, format = "%j"))
    hist_temp_min <- calc_summary(hd, var = temp_min, group_var = monthday)
    hist_temp_max <- calc_summary(hd, var = temp_max, group_var = monthday)
    hist_temp_mean <- calc_summary(hd, var = temperature, group_var = monthday)
    
    hist <- data.table(monthday = hist_temp_mean$monthday,
                       temp_min_mean  = hist_temp_min$temp_min_mean,
                       temp_min_90pct = hist_temp_min$temp_min_90pct,
                       temp_min_75pct = hist_temp_min$temp_min_75pct,
                       temp_min_25pct = hist_temp_min$temp_min_25pct,
                       temp_min_10pct = hist_temp_min$temp_min_10pct,
                       temp_max_mean  = hist_temp_max$temp_max_mean,
                       temp_max_90pct = hist_temp_max$temp_max_90pct,
                       temp_max_75pct = hist_temp_max$temp_max_75pct,
                       temp_max_25pct = hist_temp_max$temp_max_25pct,
                       temp_max_10pct = hist_temp_max$temp_max_10pct,
                       temp_mean_mean  = hist_temp_mean$temperature_mean,
                       temp_mean_90pct = hist_temp_mean$temperature_90pct,
                       temp_mean_75pct = hist_temp_mean$temperature_75pct,
                       temp_mean_25pct = hist_temp_mean$temperature_25pct,
                       temp_mean_10pct = hist_temp_mean$temperature_10pct)
    hist$date <- as.Date(paste0(this_year, paste0("-", hist$monthday)))
    hist <- hist[order(as.Date(hist$date, format="%Y-%m-%d")),]
    hist
    
  })
  
  temp_fut <- reactive({
    fd <- summarize_future_over_day()
    #fd2 <- fd[year >= input$future_range - 15 & year <= input$future_range + 15]
    fd$day <- as.numeric(strftime(fd$date, format = "%j"))
    #fd <- fd2
    
    
    fd2040 <- fd[2025 <= year & year <= 2055]
    fd2060 <- fd[2045 <= year & year <= 2075]
    fd2080 <- fd[2065 <= year & year <= 2095]
    
    fut_temp_min <- calc_summary(fd2040, var = temp_min, group_var = monthday)
    fut_temp_max <- calc_summary(fd2040, var = temp_max, group_var = monthday)
    fut_temp_mean <- calc_summary(fd2040, var = temperature, group_var = monthday)
    fut2040 <- data.table(
      monthday = fut_temp_mean$monthday,
      temp_min_mean  = fut_temp_min$temp_min_mean,
      temp_min_90pct = fut_temp_min$temp_min_90pct,
      temp_min_75pct = fut_temp_min$temp_min_75pct,
      temp_min_25pct = fut_temp_min$temp_min_25pct,
      temp_min_10pct = fut_temp_min$temp_min_10pct,
      temp_max_mean  = fut_temp_max$temp_max_mean,
      temp_max_90pct = fut_temp_max$temp_max_90pct,
      temp_max_75pct = fut_temp_max$temp_max_75pct,
      temp_max_25pct = fut_temp_max$temp_max_25pct,
      temp_max_10pct = fut_temp_max$temp_max_10pct,
      temp_mean_mean  = fut_temp_mean$temperature_mean,
      temp_mean_90pct = fut_temp_mean$temperature_90pct,
      temp_mean_75pct = fut_temp_mean$temperature_75pct,
      temp_mean_25pct = fut_temp_mean$temperature_25pct,
      temp_mean_10pct = fut_temp_mean$temperature_10pct)
    
    fut2040$date <- as.Date(paste0(this_year, paste0("-", fut2040$monthday)), format = "%Y-%m-%d")
    setkey(fut2040, date)
    fut2040$year <- year(fut2040$date)
    fut_temp_min <- calc_summary(fd2060, var = temp_min, group_var = monthday)
    fut_temp_max <- calc_summary(fd2060, var = temp_max, group_var = monthday)
    fut_temp_mean <- calc_summary(fd2060, var = temperature, group_var = monthday)
    fut2060 <- data.table(
      monthday = fut_temp_mean$monthday,
      temp_min_mean  = fut_temp_min$temp_min_mean,
      temp_min_90pct = fut_temp_min$temp_min_90pct,
      temp_min_75pct = fut_temp_min$temp_min_75pct,
      temp_min_25pct = fut_temp_min$temp_min_25pct,
      temp_min_10pct = fut_temp_min$temp_min_10pct,
      temp_max_mean  = fut_temp_max$temp_max_mean,
      temp_max_90pct = fut_temp_max$temp_max_90pct,
      temp_max_75pct = fut_temp_max$temp_max_75pct,
      temp_max_25pct = fut_temp_max$temp_max_25pct,
      temp_max_10pct = fut_temp_max$temp_max_10pct,
      temp_mean_mean  = fut_temp_mean$temperature_mean,
      temp_mean_90pct = fut_temp_mean$temperature_90pct,
      temp_mean_75pct = fut_temp_mean$temperature_75pct,
      temp_mean_25pct = fut_temp_mean$temperature_25pct,
      temp_mean_10pct = fut_temp_mean$temperature_10pct)
    
    fut2060$date <- as.Date(paste0(this_year, paste0("-", fut2060$monthday)), format = "%Y-%m-%d")
    setkey(fut2060, date)
    fut2060$year <- year(fut2060$date)
    fut_temp_min <- calc_summary(fd2080, var = temp_min, group_var = monthday)
    fut_temp_max <- calc_summary(fd2080, var = temp_max, group_var = monthday)
    fut_temp_mean <- calc_summary(fd2080, var = temperature, group_var = monthday)
    fut2080 <- data.table(
      monthday = fut_temp_mean$monthday,
      temp_min_mean  = fut_temp_min$temp_min_mean,
      temp_min_90pct = fut_temp_min$temp_min_90pct,
      temp_min_75pct = fut_temp_min$temp_min_75pct,
      temp_min_25pct = fut_temp_min$temp_min_25pct,
      temp_min_10pct = fut_temp_min$temp_min_10pct,
      temp_max_mean  = fut_temp_max$temp_max_mean,
      temp_max_90pct = fut_temp_max$temp_max_90pct,
      temp_max_75pct = fut_temp_max$temp_max_75pct,
      temp_max_25pct = fut_temp_max$temp_max_25pct,
      temp_max_10pct = fut_temp_max$temp_max_10pct,
      temp_mean_mean  = fut_temp_mean$temperature_mean,
      temp_mean_90pct = fut_temp_mean$temperature_90pct,
      temp_mean_75pct = fut_temp_mean$temperature_75pct,
      temp_mean_25pct = fut_temp_mean$temperature_25pct,
      temp_mean_10pct = fut_temp_mean$temperature_10pct)
    
    fut2080$date <- as.Date(paste0(this_year, paste0("-", fut2080$monthday)), format = "%Y-%m-%d")
    setkey(fut2080, date)
    fut2080$year <- year(fut2080$date)
    list(fut2040, fut2060, fut2080)
  })
  
  temp_hist_frost <- reactive({
    hd <- summarize_historical_over_day()
    hd$day <- as.numeric(strftime(hd$date, format = "%j"))
    
    ####### GET LAST FROST DATA #######
    last_frost <- hd[last_frost_day == 1, 
                     .(last_frost = as.integer(strftime(date, format="%j")),
                       date = as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")), 
                     by=year]
    setkey(last_frost, date) 
    
    # ####### GET FIRST FROST DATA #######    
    first_frost <- hd[first_frost_day == 1, 
                      .(first_frost = as.integer(strftime(date, format="%j")),
                        date = as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")), 
                      by=year]
    setkey(first_frost, date)  
    list(last_frost, first_frost)
  })
  
  temp_fut_frost <- reactive({
    fd <- summarize_future_over_day()
    
    ####### GET FUTURE LAST FROST DATA #######
    frost_data <- fd[last_frost_day == 1, 
                     .(last_frost = as.integer(strftime(date, format="%j")),
                       date = as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")), 
                     by=year]
    setkey(frost_data, date)
    
    l2040 <- subset(frost_data, 2025 <= year & year <= 2055)
    l2060 <- subset(frost_data, 2045 <= year & year <= 2075)
    l2080 <- subset(frost_data, 2065 <= year & year <= 2095)
    ####### GET FUTURE FIRST FROST DATA #######    
    frost_data <- fd[first_frost_day == 1, 
                     .(first_frost = as.integer(strftime(date, format="%j")),
                       date = as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")), 
                     by=year]
    setkey(frost_data, date)
    
    f2040 <- subset(frost_data, 2025 <= year & year <= 2055)
    f2060 <- subset(frost_data, 2045 <= year & year <= 2075)
    f2080 <- subset(frost_data, 2065 <= year & year <= 2095)
    
    list(l2040, l2060, l2080, 
         f2040, f2060, f2080)
  })
  
  temp_hist_heat_risk <- reactive({ 
    heat_risk <- (input$heat_risk_range - 32)*(5/9) #convert to celcius
    hd <- summarize_historical_over_day()
    hd$day <- as.numeric(strftime(hd$date, format = "%j"))
    hd <- hd[order(as.Date(hd$date, format="%Y-%m-%d")),]
    hd <- hd[monthday != "02-29"]
    
    hd$risk_day = 0
    
    hd[heat_risk < temp_max]$risk_day = 1
    hd[heat_risk >=  temp_max]$risk_day = 0
    
    hd <- hd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  date = date,
                  risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    
    hd <- hd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    hd$date <- as.Date(paste0(this_year, paste0("-", hd$monthday)), format = "%Y-%m-%d")
    
    hd <- rbind(hd[monthday == "01-15"], hd[monthday == "02-15"], hd[monthday == "03-15"], hd[monthday == "04-15"], hd[monthday == "05-15"],
                hd[monthday == "06-15"], hd[monthday == "07-15"], hd[monthday == "08-15"], hd[monthday == "09-15"], hd[monthday == "10-15"], 
                hd[monthday == "11-15"], hd[monthday == "12-15"])
    
    hist <- hd[order(as.Date(hd$date, format="%Y-%m-%d")),]
  })
  
  temp_fut_heat_risk <- reactive({ 
    heat_risk <- (input$heat_risk_range - 32)*(5/9)
    fd <- summarize_future_over_day()
    fd$day <- as.numeric(strftime(fd$date, format = "%j"))
    fd <- fd[order(as.Date(fd$date, format="%Y-%m-%d")),]
    fd <- fd[monthday != "02-29"]
    
    fd$risk_day = 0
    
    fd[heat_risk < temp_max]$risk_day = 1
    fd[heat_risk >=  temp_max]$risk_day = 0
    
    fd <- fd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  date = date,
                  risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    
    fd <- fd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    fd$date <- as.Date(paste0(this_year, paste0("-", fd$monthday)), format = "%Y-%m-%d")
    
    fd <- rbind(fd[monthday == "01-15"], fd[monthday == "02-15"], fd[monthday == "03-15"], fd[monthday == "04-15"], fd[monthday == "05-15"],
                fd[monthday == "06-15"], fd[monthday == "07-15"], fd[monthday == "08-15"], fd[monthday == "09-15"], fd[monthday == "10-15"], 
                fd[monthday == "11-15"], fd[monthday == "12-15"])
    
    fut <- fd[order(as.Date(fd$date, format="%Y-%m-%d")),]
    
    fut3_40 <- fut[2040 - 15 < year & year <= 2040 + 15]
    fut3_60 <- fut[2060 - 15 < year & year <= 2060 + 15]
    fut3_80 <- fut[2080 - 15 < year & year <= 2080 + 15]
    
    list(fut3_40, fut3_60, fut3_80)
  })
  
  temp_hist_frost_risk <- reactive({
    frost_risk <- 0
    hd <- summarize_historical_over_day()
    hd$day <- as.numeric(strftime(hd$date, format = "%j"))
    hd <- hd[order(as.Date(hd$date, format="%Y-%m-%d")),]
    hd <- hd[monthday != "02-29"]
    hd$risk_day = 0
    
    hd[temp_min < frost_risk]$risk_day = 1
    hd[frost_risk <=  temp_min]$risk_day = 0
    
    hd <- hd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  date = date,
                  risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    
    hd <- hd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    hd$date <- as.Date(paste0(this_year, paste0("-", hd$monthday)), format = "%Y-%m-%d")
    
    hd <- rbind(hd[monthday == "01-15"], hd[monthday == "02-15"], hd[monthday == "03-15"], hd[monthday == "04-15"], hd[monthday == "05-15"],
                hd[monthday == "06-15"], hd[monthday == "07-15"], hd[monthday == "08-15"], hd[monthday == "09-15"], hd[monthday == "10-15"], 
                hd[monthday == "11-15"], hd[monthday == "12-15"])
    
    hist <- hd[order(as.Date(hd$date, format="%Y-%m-%d")),]
  })
  
  temp_fut_frost_risk <- reactive({
    frost_risk <- 0
    fd <- summarize_future_over_day()
    fd$day <- as.numeric(strftime(fd$date, format = "%j"))
    fd <- fd[order(as.Date(fd$date, format="%Y-%m-%d")),]
    fd <- fd[monthday != "02-29"]
    fd$risk_day = 0
    
    fd[temp_min < frost_risk]$risk_day = 1
    fd[frost_risk <=  temp_min]$risk_day = 0
    
    fd <- fd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  date = date,
                  risk_count = cumsum(risk_day)), by=list(year(date), month(date))]
    
    fd <- fd[,  .(monthday = monthday,
                  year = year,
                  month = month,
                  risk_count = max(risk_count)), by=list(year(date), month(date))]
    
    fd$date <- as.Date(paste0(this_year, paste0("-", fd$monthday)), format = "%Y-%m-%d")
    
    fd <- rbind(fd[monthday == "01-15"], fd[monthday == "02-15"], fd[monthday == "03-15"], fd[monthday == "04-15"], fd[monthday == "05-15"],
                fd[monthday == "06-15"], fd[monthday == "07-15"], fd[monthday == "08-15"], fd[monthday == "09-15"], fd[monthday == "10-15"], 
                fd[monthday == "11-15"], fd[monthday == "12-15"])
    
    fut <- fd[order(as.Date(fd$date, format="%Y-%m-%d")),]
    
    fut3_40 <- fut[2040 - 15 < year & year <= 2040 + 15]
    fut3_60 <- fut[2060 - 15 < year & year <= 2060 + 15]
    fut3_80 <- fut[2080 - 15 < year & year <= 2080 + 15]
    
    list(fut3_40, fut3_60, fut3_80)
  })
  
  temp_pd <- reactive({
    pd <- summarize_present_over_day()
    pd
  })
  
  temp_gefs <- reactive({
    gefs <- select_gefs()[, 
                          .(temp_mean = mean(temp),
                            temp_max = mean(temp_max),
                            temp_min = mean(temp_min)), 
                          by=list(date)]
    gefs <- gefs[order(as.Date(gefs$date, format="%Y-%m-%d")),]
    gefs
  })
  
  temp_nmme <- reactive({
    hy <- summarize_historical_over_year()
    nmme1 <- select_nmme()
    nmme1[, `:=`(year = as.integer(format(date, "%y")))]
    nmme1 <- subset(nmme1, year == format(Sys.Date(), "%y"))
    nmme1[, monthday := format(date, "%m-%d")]
    nmme2 <- merge(nmme1, hy[, c("monthday", "temp_mean"), with=FALSE], by="monthday")
    nmme2[, temp := temp + temp_mean]
    nmme2[, gdd := calc_gdd(temp), by=list(date, ens)]
    
    nmme <- nmme2[, .(temp_mean = mean(temp, na.rm=T),
                      temp_min = min(temp, na.rm=T),
                      temp_max = max(temp, na.rm=T)), 
                  by=list(date)]
    
    nmme <- nmme[order(as.Date(nmme$date, format="%Y-%m-%d")),]
  })
  
  ########### SUMMARIZE FUNCTIONS ###########
  summarized_county <- reactive({
    summarize_county(select_nass(), subset_crop()$name, year, yield)
  })
  
  summarize_future_over_day = reactive({
    planting_date <- isolate(input$planting_date_longterm)
    maturity_range <- isolate(input$maturity_range_longterm)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity * (1 + maturity_range/100)
    
    dt <- summarizeOverDay(select_future(), cp, gdd_start_date = planting_date)
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  summarize_future_over_day2 = reactive({
    planting_date <- isolate(input$planting_date_longterm2)
    maturity_range <- isolate(input$maturity_range_longterm2)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity * (1 + maturity_range/100)
    
    dt <- summarizeOverDay(select_future(), cp, gdd_start_date = planting_date)
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  summarize_future_over_month = reactive({
    df <- summarize_future_over_day()[
      ,
      .(cgdd = max(cgdd, na.rm=T),
        precip = sum(precipitation),
        temp_max = mean(temp_max),
        temp_mean = mean(temperature),
        temp_min = mean(temp_min)
      ),
      
      c("month", "year")
      ]
    df
  })
  
  summarize_future_over_year = reactive({
    dt <- summarizeOverYear(
      summarize_future_over_day())[,
        date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")
        ]
    setkey(dt, date)
    if(leap_year(this_year)) res <- dt[monthday != "02-29"]
    else res <- dt
    res
  })
  summarize_future_over_year2 = reactive({
    dt <- summarizeOverYear(
      summarize_future_over_day2())[,
        date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")
        ]
    setkey(dt, date)
    if(leap_year(this_year)) res <- dt[monthday != "02-29"]
    else res <- dt
    res
  })
  
  summarize_historical_over_day = reactive({
    planting_date <- isolate(input$planting_date_longterm)
    maturity_range <- isolate(input$maturity_range_longterm)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity * (1 + maturity_range/100)
    
    dt <- summarizeOverDay(select_historical(), cp, gdd_start_date = planting_date)
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  summarize_historical_over_day2 = reactive({
    planting_date <- isolate(input$planting_date_longterm2)
    maturity_range <- isolate(input$maturity_range_longterm2)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity * (1 + maturity_range/100)
    
    dt <- summarizeOverDay(select_historical(), cp, gdd_start_date = planting_date)
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  
  summarize_historical_over_month = reactive({
    dt <- summarize_historical_over_day()[
      ,
      .(cgdd = max(cgdd, na.rm=T),
        precip = sum(precipitation),
        temp_max = mean(temp_max),
        temp_mean = mean(temperature),
        temp_min = mean(temp_min)
      ),
      c("month", "year")
      ]
    dt
  })
  
  summarize_historical_over_year = reactive({
    dt <- summarizeOverYear(
      summarize_historical_over_day())[,
        date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")
        ]
    setkey(dt, date)
    if(leap_year(this_year)) res <- dt[monthday != "02-29"]
    else res <- dt
    res
  })
  
  summarize_historical_over_year2 = reactive({
    dt <- summarizeOverYear(
      summarize_historical_over_day2())[,
        date := as.Date(paste0(this_year, "-", monthday), format = "%Y-%m-%d")
        ]
    setkey(dt, date)
    if(leap_year(this_year)) res <- dt[monthday != "02-29"]
    else res <- dt
    res
  })
  
  summarize_present_over_day = reactive({
    planting_date <- isolate(input$planting_date_longterm)
    maturity_range <- isolate(input$maturity_range_longterm)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity + cp$maturity * maturity_range/100
    
    dt <- summarizeOverDay(select_reacch(), cp, gdd_start_date = planting_date)
    dt[date < today]
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  summarize_present_over_day2 = reactive({
    planting_date <- isolate(input$planting_date_longterm2)
    maturity_range <- isolate(input$maturity_range_longterm2)
    
    cp <- subset_crop()
    cp$maturity = cp$maturity + cp$maturity * input$maturity_range/100
    dt <- summarizeOverDay(select_reacch(), cp, gdd_start_date = planting_date)
    dt[date < today]
    if(leap_year(this_year)) dt <- dt[monthday != "02-29"]
    dt
  })
  
  summarize_present_over_month = reactive({
    df <- summarize_present_over_day()[,
                                       .(cgdd = max(cgdd, na.rm=T),
                                         precip = sum(precipitation)),
                                       month]
    df
  })
  
  ########### SELECTION FUNCTIONS ###########
  
  select_cfs <- reactive({
    loc <- select_location()
    cfs <- read_forecast(lat = loc$latitude,
                         lon = loc$longitude,
                         model = "cfs",
                         today = today,
                         data_dir = paste0(data_dir, "cfs/"))
    cfs
  })
  
  select_codling <- reactive({
    loc <- select_location()
    
    location_lat <- input$map_shape_click$lat
    location_lng <- input$map_shape_click$lng
    
    data_sets <- NULL
    
    file_string <- paste0(data_dir, "codling/bcc-csm1-1-m/CM_", location_lat, "_", location_lng)
    if (file.exists(file_string))
    {
      codling_moth_bcc <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_bcc$model <- "bcc-csm1-1-m"
      codling_moth_bcc$modelagg <- "future"
      data_sets <- codling_moth_bcc
      
      file_string <- paste0(data_dir, "codling/BNU-ESM/CM_", location_lat, "_", location_lng)
      codling_moth_bnu <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_bnu$model <- "BNU-ESM"
      codling_moth_bnu$modelagg <- "future"
      data_sets <- rbind(data_sets, codling_moth_bnu)
      
      file_string <- paste0(data_dir, "codling/CanESM2/CM_", location_lat, "_", location_lng)
      codling_moth_can <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_can$model <- "CanESM2"
      codling_moth_can$modelagg <- "future"
      data_sets <- rbind(data_sets, codling_moth_can)
      
      file_string <- paste0(data_dir, "codling/CNRM-CM5/CM_", location_lat, "_", location_lng)
      codling_moth_cnrm <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_cnrm$model <- "CNRM-CM5"
      codling_moth_cnrm$modelagg <- "future"
      data_sets <- rbind(data_sets, codling_moth_cnrm)
      
      file_string <- paste0(data_dir, "codling/GFDL-ESM2G/CM_", location_lat, "_", location_lng)
      codling_moth_gfdl_esm2g <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_gfdl_esm2g$model <- "GFDL-ESM2G"
      codling_moth_gfdl_esm2g$modelagg <- "future"
      data_sets <- rbind(data_sets, codling_moth_gfdl_esm2g)
      
      file_string <- paste0(data_dir, "codling/GFDL-ESM2M/CM_", location_lat, "_", location_lng)
      codling_moth_gfdl_esm2m <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_gfdl_esm2m$model <- "GFDL-ESM2M"
      codling_moth_gfdl_esm2m$modelagg <- "future"
      data_sets <- rbind(data_sets, codling_moth_gfdl_esm2m)
      
      file_string <- paste0(data_dir, "codling/hist/CM_", location_lat, "_", location_lng)
      codling_moth_hist <- read.table(file_string, header=TRUE, sep=",")
      codling_moth_hist$model <- "historical"
      codling_moth_hist$modelagg <- "historical"
      data_sets <- rbind(data_sets, codling_moth_hist)
    }
    
    ds1 <- data_sets[which(data_sets$year > 2024 & data_sets$year < 2056 & data_sets$modelagg == "future"),]
    ds1$yearquantile <- "2040s"
    ds2 <- data_sets[which(data_sets$year > 2044 & data_sets$year < 2076 & data_sets$modelagg == "future"),]
    ds2$yearquantile <- "2060s"
    ds3 <- data_sets[which(data_sets$year > 2064 & data_sets$modelagg == "future"),]
    ds3$yearquantile <- "2080s"
    ds4 <- data_sets[which(data_sets$year < 2008 & data_sets$modelagg == "historical"),]
    ds4$yearquantile <- "Historical"
    
    data_sets <- ds1
    data_sets <- rbind(data_sets, ds2)
    data_sets <- rbind(data_sets, ds3)
    data_sets <- rbind(data_sets, ds4)
    
    return (data_sets)
  })
  
  select_future <- reactive({
    loc <- select_location()
     model <- input$mod
     rcp <- input$rcp
    read_future(lat = loc$latitude, lon = loc$longitude, years = future_year_range, model=model, rcp=rcp)
  })
  
  select_gefs <- reactive({
    loc <- select_location()
    read_forecast(lat = loc$latitude,
                  lon = loc$longitude,
                  model = "gefs",
                  today = today,
                  data_dir = paste0(data_dir, "gefs/"))
  })
  
  select_historical <- reactive({ #on adjusting the range of years
    loc <- select_location()
    read_historical(lat = loc$latitude, lon = loc$longitude, years = historical_year_range)
  })
  
  select_location <- reactive({
    if(!is.null(input$map_shape_click)) {
      event = input$map_shape_click
      res <- list(latitude=event$lat, longitude=event$lng)
    }
    else if (!is.null(input$map_shape_mouseover))
    {
      event = input$map_shape_mouseover
      res <- list(latitude=event$lat, longitude=event$lng)
    }
    else {
      res <- NULL      
    }
    res
  })
  
  select_nass <- reactive({
    loc <- select_location()
    nass <- data.table(read_nass.latlon(loc$latitude, loc$longitude))
    names(nass) <- str_to_lower(names(nass))
    nass[commodity_desc=="WHEATWINTER"]$commodity_desc = "WHEAT, WINTER"
    nass[commodity_desc=="WHEATSPRING, (EXCL DURUM)"]$commodity_desc = "WHEAT, SPRING, (EXCL DURUM)"
    nass[commodity_desc=="BEANSDRY EDIBLE"]$commodity_desc = "BEANS, DRY EDIBLE, CHICKPEAS"
    nass[commodity_desc=="HAYALFALFA"]$commodity_desc = "HAY, ALFALFA"
    nass[commodity_desc=="HAY(EXCL ALFALFA)"]$commodity_desc = "HAY, (EXCL ALFALFA)"
    nass
  })
  
  select_nmme <- reactive({
    loc <- select_location()
    dt <- read_forecast(loc$latitude, 
                        loc$longitude,
                        model="nmme",
                        today = today,
                        data_dir = paste0(data_dir, "nmme/"))
    dt[, monthday := format(date, "%m-%d")]
    dt[date>=today & date <= last_day_of_year]
  })
  
  select_reacch <- reactive({
    loc <- select_location()
    dt <- read_present(loc$latitude,
                       loc$longitude,
                       today = today)
  })
  
  spatial_future <- reactive({
    dt <- create_distance_dt(summarize_future_over_day(), varlist = input$analogue_vars)
    dt
  })
  
  subset_crop <- reactive({
    #input$gddButton
    input$gddButton2
    crop_name <- isolate(input$crop_name_longterm)
    crop_name_longterm <- isolate(input$crop_name_longterm)
    #crop_name <- if(is.null(crop_name)) crop_name_longterm
    filter(crops, name == crop_name)
  })
  
  subset_future <- reactive({
    select_future()[]
  })
  
  subset_historical <- reactive({
    select_historical()[year >= input$historical_range[1] &
                          year <= input$historical_range[2]]
  })
  
  subset_spatial_historical <- reactive({
    nh <- names(spatial_historical)
    vars <- c("year", "lat", "lon")
    for(i in input$analogue_vars) {
      this_var <- nh[grepl(i, nh)]
      vars <- c(vars, this_var)
    }
    spatial_historical[, vars, with=FALSE]
  })
  
  subset_nass_yields <- reactive({
    crop <- subset_crop()
    nass <- select_nass()
    
    #Fix unit based on crop
    if (crop$nass_name == "BEANS, DRY EDIBLE, CHICKPEAS") unit_d <- "LB / ACRE" else unit_d <- "BU / ACRE"
    nass_yields <- nass[commodity_desc == crop$nass_name & 
                          unit_desc == unit_d &
                          year >= min(historical_years), 
                        c("year", "yield"), 
                        with = FALSE]
  })
  
  time_historical <- reactive({
    dt <- create_distance_dt(summarize_historical_over_day(), varlist = c("cum_gdd"))
    dt
  })
  
  getTimeFrame <- reactive({
    if (input$timeFrame == "Historical")
      return (summarize_historical_over_year())
    else if (input$timeFrame == "Future")
      return (summarize_future_over_year())
  })
  
  ###################################################################################################
  #Observers
  ###################################################################################################
  
  observe({
    if (input$tileSelect == "Satellite"){
      leafletProxy("map") %>%
        addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                 layerId = "Satellite")
      
      # Update layer id 
      layer_id = 0;  
    }
    else if (input$tileSelect == "Topographic"){
      leafletProxy("map") %>%
        addTiles(urlTemplate = "http://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                 layerId = "Topographic")
      
      # Update layer id 
      layer_id = 1;
    }
    else if (input$tileSelect == "Basic"){
      leafletProxy("map") %>%
        addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                 layerId = "Basic")
      
      # Update layer id 
      layer_id = 2;
    }
  })
  
  ############### edit here
  observe({
    event <- input$map_shape_click
    if (is.null(event))
      return()
    if (is.null(event$group))
      return()
    if (event$group == "locations")
      leafletProxy("map") %>% clearPopups()
    
    selected_crops <- subset(crops, name %in% availableCrops(event$lat, event$lng, crops, crop_locations))
    default_crop_name <- head(selected_crops, 1)
    
    updateSelectInput(session, "crop_name", "Crops", selected_crops$name, selected = default_crop_name)
    updateSelectInput(session, "crop_name_longterm", "Crops", selected_crops$name, selected = default_crop_name)
    
    toggleModal(session, modalId="graphs", toggle = "open")
  })
  
  
  ### LINK PLOT PARAMETERS ####
  observe({
    updateSliderInput(session, "maturity_range_longterm", "Default Maturity % Offset:", -30, 30, value=input$maturity_range)
  })
  
  observe({
    updateSliderInput(session, "maturity_range", "Default Maturity % Offset:", -30, 30, value=input$maturity_range_longterm)
  })
  
  observe({
    updateDateInput(session, "planting_date_longterm", label="Default Planting Date", value = input$planting_date)
  })
  
  observe({
    updateDateInput(session, "planting_date", label="Default Planting Date", value = input$planting_date_longterm)
  })
  
  observe({
    updateSelectInput(session, "crop_name_longterm", "Crops:", selected_crops$name, selected = input$crop_name)
  })
  
  observe({
    updateSelectInput(session, "crop_name", "Crops:", selected_crops$name, selected = input$crop_name_longterm)
  })
  
  observe({
    input$cropSelectAll
    updateSelectInput(session, "cropSelect", "Locations with:", crop_names, selected=crop_names)
    
  })
  ######  
  observe({
    if (is.null(input$cropSelect))
    {
      leafletProxy("map") %>%
        clearGroup("locations")
      return();
    }
    
    crop_id <- subset(crops, name %in% input$cropSelect)$id
    selected_locations <- subset(crop_locations, as.numeric(cropid) %in% as.numeric(crop_id))
    selected_locations <- selected_locations[!duplicated(selected_locations[,-4]), ]
    
    pal = colorBin(palette = "Reds", domain = as.numeric(selected_locations$num_crops), bins = 8, pretty=FALSE)
    
    leafletProxy("map") %>%
      clearGroup("locations") %>%
      addCircles(data = selected_locations, lng = ~longitude, lat = ~latitude,
                 radius=r,
                 stroke=FALSE,
                 fillOpacity=0.6,
                 color = ~pal(as.numeric(num_crops)),
                 group = "locations") %>%
      
      setView(lng = -117, lat = 45, zoom = 6)
    
  })
  
  observe({
    event <- input$map_shape_mouseover
    if (is.null(event)) { return() } 
    if (is.null(event$group)) { return() } 
    if (event$group == "locations") 
    {
      irr <- irrigation[lat == event$lat & long == event$lng]
      content <- sprintf("Coordinates: %f, %f", event$lat, event$lng)
      
      if (0 < nrow(irr))
      {
        content <- paste0(content, sprintf("<br/>County: %s", head(irr$countyname, 1)))
        content <- paste0(content, paste0("<br/>Irrigated Crops: ", paste(irr[irrtype != 0]$cropname, collapse = ", ")))
        content <- paste0(content, paste0("<br/>Non-irrigated Crops: ", paste(irr[irrtype == 0]$cropname, collapse = ", ")))
      }
      
      leafletProxy("map") %>% addPopups(lat = event$lat + 0.05, lng = event$lng, content)
    }
  })
  
  observe({
    event <- input$map_shape_mouseout
    leafletProxy("map") %>% clearPopups()
  })
  
})



############# ET FUNCTIONS ##################################################################

# Functions 

#Parameters


##initial --------------------

##  Lat   - degrees -   latitude
##  DOY   -   NA    -   year day 
##  Tmax  -   C     -   Max temp 
##  Tmin  -   C     -   Min temp
##  RS    - Mj/m2/d -   daily solar radiation in Megajoule per square meter
##  RHmax -   %     -   Max relative humidity
##  RHmin -   %     -   Min relative humidity 
##  Uz    -   m/s   -   Wind speed 
##  SH    -   m     -   screening height, static in spreadsheet (2) 
##  rc    -   d/m   -   surface resistence, static in spreadsheet (0.00081)
##  Elev  -   m     -   Elevation, static in spreasheet (100) 

SatVP <- function(Temp){
  0.6108 * exp(17.27 * Temp / (Temp + 237.3))
}

VP <- function(esTmax, esTmin, RHmax, RHmin){
  (esTmin * RHmax / 100 + esTmax * RHmin / 100) / 2
}

VPD <- function(esTmax, esTmin, ea){
  (esTmax + esTmin) / 2 - ea
}

#for some reason VBA code uses this. Why? idk. I used built in acos() function. 
#f_arccos <- function(x){ 
#  atan(-x / sqrt(-x * x + 1)) + 2 * atan(1)
#}

PotRad <- function(Lat, DOY){
  Pi = 3.14159
  Solar_Constant = 118.08
  Lat_Rad = Lat * Pi / 180
  dr = 1 + 0.033 *  cos(2 * Pi * DOY / 365)
  SolDec = 0.409 * sin(2 * Pi * DOY / 365 - 1.39)
  SunsetHourAngle = acos(-tan(Lat_Rad) * tan(SolDec))
  Term = SunsetHourAngle * sin(Lat_Rad) * sin(SolDec) + cos(Lat_Rad) * cos(SolDec) * sin(SunsetHourAngle)
  Solar_Constant * dr * Term / Pi
}


NetRad <- function(ER, Rs, ea, Tmax, Tmin){
  #Calculate shortwave net radiation
  albedo = 0.23
  Rns = (1 - albedo) * Rs
  
  #Calculate cloud factor
  F_Cloud = 1.35 * (Rs / (ER * 0.75)) - 0.35
  
  #Calculate humidity factor
  F_Hum = (0.34 - 0.14 * sqrt(ea))
  
  #Calculate Isothermal LW net radiation
  LWR = 4.903E-09 * ((Tmax + 273) ^ 4 + (Tmin + 273) ^ 4) / 2
  Rnl = LWR * F_Cloud * F_Hum
  
  #Calculate Rn
  Rns - Rnl
  
}


AeroRes <- function(Uz, z){
  if(z == 2){
    U <- Uz
  }  else {
    U <- Uz * (4.87 / (log(67.8 * z - 5.42)))
  }
  U2 = U * 86400 #Convert to m/day
  d = 0.08
  zom = 0.01476
  zoh = 0.001476
  zm = 2
  zh = 2
  VK = 0.41
  Term1 = log((zm - d) / zom)
  term2 = log((zh - d) / zoh)
  Term1 * term2 / (VK * VK * U2)
}

DT <- function(Tmean, esTmean){
  4098 * esTmean / ((Tmean + 237.3) ^ 2)
}

Lambda <- function(Tmean){
  2.501 - 0.002361 * Tmean
}


Gamma <- function(Lambda, Elev){
  Cp = 0.001013
  P = 101.3 * ((293 - 0.0065 * Elev) / 293) ^ 5.26
  Cp * P / (0.622 * Lambda)
}

ETRadTerm <- function(Delta, Gamma, Lambda, Rn, rc, ra){
  ETRTerm = Delta * Rn / (Delta + Gamma * (1 + rc / ra))
  ETRTerm / Lambda
}


ETAeroTerm <- function(Delta, Gamma, Lambda, rc, ra, VPD, Tmean, Elev){
  Cp = 0.001013
  P = 101.3 * ((293 - 0.0065 * Elev) / 293) ^ 5.26
  Tkv = 1.01 * (Tmean + 273)
  AirDensity = 3.486 * P / Tkv
  VolHeatCap = Cp * AirDensity
  ETAeroTerm = (VolHeatCap * VPD / ra) / (Delta + Gamma * (1 + rc / ra))
  ETAeroTerm / Lambda
}

#my own function mimicing total ET 
ETtotal <- function(ETRad, ETAero){
  sum(ETRad, ETAero)
}



## adding ET Column
addET <- function(dt, ScreeningHeight =2, SurfaceResistance=0.00081, lat, elevation) {
  
  ET <- c()
  for (i in 1:nrow(dt)){ 
    Lat   <- lat
    DOY   <- dt$yday[i] 
    Tmax  <- dt$temp_max[i]
    Tmin  <- dt$temp_min[i]
    Rs    <- dt$SRAD[i] 
    RHmax <- dt$RMAX[i]
    RHmin <- dt$RMIN[i]
    Uz    <- dt$windspeed[i]
    SH    <- ScreeningHeight   
    rc    <- SurfaceResistance
    Elev  <- elevation 
    
    #    source ET functions (put in server environment, or in package if you can)
    
    #generate intermediates
    Tmean   <- mean(c(Tmax, Tmin))
    esTmax  <- SatVP(Tmax)
    esTmin  <- SatVP(Tmin)
    esTmean <- SatVP(Tmean)
    ea      <- VP(esTmax, esTmin, RHmax, RHmin)
    vpd     <- VPD(esTmax, esTmin, ea)
    PRad    <- PotRad(Lat, DOY)
    Rn      <- NetRad(PRad,Rs, ea, Tmax, Tmin) 
    ra      <- AeroRes(Uz, SH)
    Delt    <- DT(Tmean, esTmean)
    Lamb    <- Lambda(Tmean)
    Gamm    <- Gamma(Lamb, Elev)
    ETRad   <- ETRadTerm(Delt, Gamm, Lamb, Rn, rc, ra)
    ETAer   <- ETAeroTerm(Delt, Gamm, Lamb, rc, ra, vpd, Tmean, Elev)
    ETtot   <- ETtotal(ETRad, ETAer)
    
    #     append column with ETtot
    
    ET <- c(ET,ETtot) 
  }
  
  dt <- cbind(dt, ET)
  
}


