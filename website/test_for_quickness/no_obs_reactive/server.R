#
# from 
# https://shiny.rstudio.com/reference/shiny/latest/updateSelectInput.html
#
# useful in goddamn update reactivity thing:
# https://stackoverflow.com/questions/28379937/change-selectize-choices-but-retain-previously-selected-values
#


shinyServer(function(input, output, session) {  
  current_selection <- reactiveVal(NULL)

  observeEvent(input$subbasins_id, {
    if (input$purpose_id == "pod"){
       plot_dt <- spatial_wtr_right
       } else {
        plot_dt <- places_of_use
    }
    curr_spatial <- plot_dt
    
    curr_spatial <- curr_spatial %>% 
                    filter(subbasin %in% input$subbasins_id) %>% 
                    data.table()

    subbasin_to_plot <- unique(curr_spatial$subbasin)
  })

  output$water_right_map <- renderLeaflet({
    target_date <- as.Date(input$cut_date)
    if (input$purpose_id == "pod"){
       plot_dt <- spatial_wtr_right
       # curr_spatial <- plot_dt
       plot_dt[, colorr := ifelse(right_date < target_date, 
                                    "#FF3333", "#0080FF")]
       } else {
      plot_dt <- places_of_use
      plot_dt <- plot_dt %>% 
                 filter(right_date > target_date)%>%
                 data.table()
    }
    #########################################################
    if (input$water_source_type == "surfaceWater") {
      curr_spatial <- plot_dt %>%
                      filter(WaRecRCWCl == "surfaceWater")%>%
                      data.table()

     } else if (input$water_source_type == "groundwater"){
      curr_spatial <- plot_dt %>%
                      filter(WaRecRCWCl == "groundwater")%>%
                      data.table()
       } else {
          curr_spatial <- data.table(plot_dt)
    }
      curr_spatial <- curr_spatial %>% 
                      filter(WRIA_NM == input$countyType_id) %>% 
                      data.table()
      
      subbasins <- sort(unique(curr_spatial$subbasin))

    observeEvent(input$subbasins_id, {
             current_selection(input$subbasins_id)
             })
    curr_s <- current_selection()
    print (curr_s)

    if (!(curr_s %in% subbasins)){
       sss <- sort(unique(curr_spatial$subbasin))
      } else {
        sss <- current_selection()
    }
    
    updateSelectInput(session, 
                      inputId = "subbasins_id", 
                      choices = subbasins,
                      selected = sss)
  #############################################
    water_map <- build_map(data_dt = curr_spatial, 
                           sub_bas = curr_s)
    water_map

  })

})

