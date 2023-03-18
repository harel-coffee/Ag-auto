#' make a gdd plot with current year forecast.
#' 
#' High level function to plot gdd, using either ggplot2 or plotly.
#' 
#' @export
#'
#' @param ... parameters passed to low level functions.
#' @param as_lines plot as lines instead of distribution.
#' @param engine rbokeh or ggplot2, determining which engine to use.
plot_gdd_forecast <- function(..., as_lines = FALSE, engine = c("plotly", "ggplot2")) {
  engine <- match.arg(engine)
  args <- list(...)
  if (!("colors" %in% names(args))) {
    args$colors = palette3
  }
  if (engine == "plotly") do.call(plot_gdd_forecast.plotly, args)
  else if (engine == "ggplot2") do.call(plot_gdd_forecast.ggplot, args)
}


#' Plot GDD with the plotly library
#' @export
#' @import plotly
#' @rdname plot_gdd_forecast
#' @return a GDD plot object made with plotly
plot_gdd_forecast.plotly <- function(crop, historical, present,
                                     gefs, nmme, stage_dates, 
                                     hist_last_frost_pct, hist_first_frost_pct, 
                                     colors) {
  
  
  
  lblank <- list(color = toRGB("white", alpha=0)) #for the bottom range of ribbons, but invisible
  fl <- list(size = "18")
  ft <- list(size = "14")
  

  yax1 <- list(anchor = "x", domain = c(0, .15), 
               showticklabels = FALSE,
               title = "",
               mirror = "ticks",
               zeroline = TRUE)

  yax3 <- list(anchor = "x", domain = c(.15, .3), dtick = 2, 
               title = "", showticklabels = FALSE,mirror = "ticks",
               zeroline = TRUE)
  
  yax2 <- list(anchor = "x", domain = c(.3, 1), 
               title = "Cumulative GDD",  showticklabels = FALSE, titlefont = fl, tickfont = ft, mirror = "ticks",
               zeroline = TRUE)

  #x axis
  xax <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              hoverformat = "%b %e",
              range(c(first_date, last_date)),
              tickfont = ft)
  
  leg <- list(x = 0,
              y = 1,
              xanchor = 'left',
              yanchor = 'left',
              orientation = 'h')
  
  xend = present[first_frost_day==1]$date
  
  if (length(xend) == 0L)
  {
    xend = paste0(year(today), "-12-31")
    txt = ""
    show = FALSE
  }
  else
  {
    txt = "Growing Season End"
    show = TRUE
  }
  
  p <- plot_ly() %>% 
  layout(
    xaxis = xax, yaxis = yax1, yaxis2 = yax2, yaxis3 = yax3,
                            shared_xaxes=FALSE,
                            legend = leg,
                            annotations = list(
                              list(xref = 'x', yref = 'y1',
                                   x = min(hist_last_frost_pct$date),
                                   y = 1,
                                   showarrow = FALSE,
                                   text = "Growing Season Start"
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(hist_first_frost_pct$date),
                                   y = 1,
                                   showarrow = FALSE,
                                   text = "Growing Season End"
                              ),
                              list(xref = 'x', yref = 'y2',
                                   x = paste0(year(today), "-7-20"),
                                   y = 6000,
                                   showarrow = FALSE,
                                   text = "Cumulative GDD"),

                              list(xref = 'x', yref = 'y2',
                                   x = present[last_frost_day==1]$date,
                                   y = 0,
                                   showarrow = TRUE,
                                   arrowhead = 2,
                                   text = "Growing Season Start"),
                              list(xref = 'x', yref = 'y2',
                                   x = xend,
                                   y = 0,
                                   showarrow = show,
                                   arrowhead = 2,
                                   text = txt),
                              list(xref = 'x', yref = 'y2',
                                   x = max(present$date),
                                   y = max(present$cgdd),
                                   ax = -25,
                                   ay = -25,
                                   showarrow = TRUE,
                                   arrowhead = 2,
                                   text = "Today"),
                              
                              list(xref = 'x', yref = 'y3',
                                   x = min(stage_dates[first_emergence_day == 1]$date),
                                   y = 1,
                                   showarrow = FALSE,
                                   xanchor = "left",
                                   text = "Emergence Dates"),
                              list(xref = 'x', yref = 'y3',
                                   x = max(stage_dates[first_maturity_day == 1]$date),
                                   y = 1,
                                   showarrow = FALSE,
                                   text = "Maturity Dates",
                                   xanchor = "center"),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 0.4,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(historical[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - historical[date == paste0("", mean(stage_dates[first_maturity_day == 1]$date))]$cgdd_mean, 2)
                                   ),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 1,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = "Post Maturity GDD"
                              )
                            )
            
         )
  
  ### HISTORICAL, FUTURE, and FORECAST DATA
  # Historical min/max
  lhist <- list(color = toRGB(colors$hist, alpha=0.3))
  p <- add_trace(p, data = historical, x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank,
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_max, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", fill = "tonexty", line = lhist, 
                 name = "Historical 0-100%", hoverinfo = "none")

 
  # Historical 2 standard deviations 
  p <- add_trace(p, data = historical, x = date, y = cgdd_10pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_90pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", fill = "tonexty", line = lhist, 
                 name = "Historical 10-90%", hoverinfo = "none")
  
  # Historical 1 standard deviation
  p <- add_trace(p, data = historical, x = date, y = cgdd_25pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_75pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", fill = "tonexty", line = lhist, 
                 name = "Historical 25-75%", hoverinfo = "none")
  
  
  
  
  # Historical mean
  lhist_mean <- list(color = toRGB(colors$hist, alpha=1))
  p <- add_trace(p, data = historical, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 line = lhist_mean,
                 text = paste("Historical Mean:", round(historical$cgdd_mean,0)),
                 name = "Historical Mean",
                 hoverinfo = "x+text")

  
  
  
  # p <- add_trace(p, y = c(-0.5, -0.5),
  #                xaxis = "x1", yaxis = "y3",
  #                line = list(color = "black"))
  
  # NMME Forecast
  lnmme <- list(color = toRGB(colors$nmme, alpha=.2))
  lnmme_mean <- list(color = toRGB(colors$nmme, alpha=.6))
  p <- add_trace(p, data = nmme, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lnmme_mean, 
                 text = paste("Long-Term Forecast:", round(nmme$cgdd_mean, 0)), 
                 name = "Long-Term Forecast (NMME) Mean", 
                 hoverinfo = "x+text") %>%
       add_trace(x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_max, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", fill = "tonexty", 
                 line = lnmme, name = "Long-Term Forecast (NMME) 0-100%", 
                 hoverinfo = "none")

  # GEFS Forecast
  lgefs <- list(color = toRGB(colors$gefs, alpha=.2))
  lgefs_mean <- list(color = toRGB(colors$gefs, alpha=.6))
  p <- add_trace(p, data = gefs, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lgefs_mean, 
                 text = paste("Short-Term Forecast:", round(gefs$cgdd_mean, 0)), 
                 name = "Short-Term Forecast (GEFS) Mean", 
                 hoverinfo = "x+text") %>%
       add_trace(x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_max, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", fill = "tonexty", line = lgefs, 
                 name = "Short-Term Forecast (GEFS) 0-100%", hoverinfo = "none")

  
  ### CROP STAGE RANGES
  p <- add_trace(p, data = stage_dates[first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$hist),
                 showlegend = FALSE, name = "Emergence Range", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 0.4, xaxis = "x1", yaxis = "y3") %>%
       add_trace(data = stage_dates[first_maturity_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$hist),
                 showlegend = FALSE, name = "Maturity Range", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 0.4, xaxis = "x1", yaxis = "y3")
  
  # PRESENT DATA
  lp <- list(color = palette3$present)
  p <- add_trace(p, data = present, x = date, y = cgdd, 
                 xaxis = "x1", yaxis = "y2",
                 line = lp, 
                 text = paste("Year-to-Date:", round(cgdd,0)), name = "Year-to-Date", hoverinfo = "x+text")

  p <- add_trace(p, data = hist_last_frost_pct, x = date,
            type = "box", orientation = "h", line = list(color = palette3$hist),
            showlegend = FALSE, hoverinfo="x+text",
            text = paste("Mean Start Date:", mean(date)),
            name = "Growing Season Start", y0 = 0.4, xaxis = "x1", yaxis = "y1")

  p <-  add_trace(p, data = hist_first_frost_pct, x = date,
              type = "box", orientation = "h", line = list(color = palette3$hist),
              showlegend = FALSE, hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)),
              name = "Growing Season End", y0 = 0.4, xaxis = "x1", yaxis = "y1")
  p
}

#' Make a GDD Plot with ggplot2.
#' 
#' @import ggplot2
#' @importFrom scales date_format
#' @export
#'  
#' @rdname plotGDD
#' @return a plot object made with ggplot2
plot_gdd_forecast.ggplot <- function(crop, historical, present, future = NULL,
                           gefs, cfs, nmme, #msy,
                           optLayers, colors) {
  
  this_year = as.integer(format(present$date[1], "%Y"))
  
  if(is.null(historical$date))
    historical$date = with(historical, as.Date(paste0(this_year, "-", monthday)))
  if(is.null(future$date) & !missing(future) & !is.null(future))
    future$date = with(future, as.Date(paste0(this_year, "-", monthday)))
  
  first_day = as.Date(paste0(this_year, "-01-01"))
  last_day = as.Date(paste0(this_year, "-12-31"))
  
  ymax <- max(c(max(historical$cgdd_max), max(present$cgdd), max(future$cgdd_max)))
  
  #Plot
  p <- ggplot(data=historical) + geom_blank() +
    
    # x axis scale
    date_scale() +
    
    #Limits
    #ylim(c(0,max(future$cgdd_max)+100)) + 
    
    # labels
    xlab("Month") + ylab("Growing Degree Days")
  
  # Historical GDD range
  add_h_title <- FALSE
  if("layerHistMinMax" %in% optLayers) {
    add_h_title <- TRUE
    p = p + geom_ribbon(data=historical, 
                        aes(ymin=cgdd_min, ymax=cgdd_max, x=date), 
                        fill=colors$hist_minmax, alpha=.3) 
  }
  if("layerHist1090pct" %in% optLayers) {
    add_h_title <- TRUE
    p = p + geom_ribbon(data=historical, 
                        aes(ymin=cgdd_10pct, ymax=cgdd_90pct, x=date), 
                        fill=colors$hist_2sd, alpha=0.5)
  }
  if("layerHist2575pct" %in% optLayers) {
    add_h_title <- TRUE
    p = p + geom_ribbon(data=historical, 
                        aes(ymin=cgdd_25pct, ymax=cgdd_75pct, x=date), 
                        fill=colors$hist_1sd, alpha=0.75)
  } 
  if("layerHistMean" %in% optLayers) {
    add_h_title = TRUE
    p = p + geom_line(data=historical, aes(y=cgdd_mean, x=date), 
                      color=colors$hist_mean, size=.7) 
  } 
  if(add_h_title) {
    p <- p + annotate("text", x = first_day, y = ymax,
                      label = "Historical Range", color = colors$hist_title,
                      hjust = 0) 
    ymax <- ymax - 100
  } 
  
  #future GDD Ranges and mean
  add_f_title = FALSE
  if("layerFutMinMax" %in% optLayers) {
    add_f_title = TRUE
    p = p + geom_ribbon(data=future, 
                        aes(ymin=cgdd_min, ymax=cgdd_max, x=date), 
                        fill=colors$fut_minmax, alpha=0.25)
  }
  if("layerFut1090pct" %in% optLayers) {
    add_f_title = TRUE
    p = p + geom_ribbon(data=future, 
                        aes(ymin=cgdd_10pct, ymax=cgdd_90pct, x=date), 
                        fill=colors$fut_2sd, alpha=0.5)
  }
  if("layerFut2575pct" %in% optLayers) {
    add_f_title = TRUE
    p = p + geom_ribbon(data=future, 
                        aes(ymin=cgdd_25pct, ymax=cgdd_75pct, x=date), 
                        fill=colors$fut_1sd, color="#FB2A1A", alpha=0.75)
    
  } 
  if("layerFutMean" %in% optLayers) {
    add_f_title = TRUE
    p = p + geom_line(data=future, aes(y=cgdd_mean, x=date), 
                      size=.7, color=colors$fut_mean)
  }
  if(add_f_title) {
    p <- p + annotate("text", x = first_day, y = ymax,
                      label = "Future Range", color = colors$fut_title,
                      hjust = 0)
    ymax <- ymax - 100
  } 
  
  # This year's gdd
  p = p + 
    geom_line(data = present, 
              aes(y=cgdd, x=date), color=colors$present, size=1) +
    annotate("text", x = first_day, y = ymax, label = "Current Year", 
             color = colors$present, hjust = 0)
  ymax <- ymax - 100
  
  #Plot stage lines
  tmpcfs <- cfs[model == factor(1), c("date", "cgdd_mean"), with=FALSE]
  setnames(tmpcfs, "cgdd_mean", "cgdd")
  stage_dat <- rbind(present[, c("date", "cgdd"), with=FALSE], 
                     tmpcfs[, c("date", "cgdd"), with=FALSE])
  setkey(stage_dat, date)
  ystage_label = 250
  if("layerMaturity" %in% optLayers && crop$maturity <= max(stage_dat$cgdd)) {
    stage_date <- historical[cgdd_max>=crop$maturity]$date[1]
    p <- addStageLayers.ggplot(p, historical, stage_dat, 
                               crop$maturity, ystage_label,
                               color = colors) +
      annotate("text", x = stage_date, y = ystage_label, 
               label = "Maturity Range", 
               hjust = 0, vjust = -2, size=4.5,
               color = "gray10")
    #ystage_label <- ystage_label + 100
  }
  if("layerSenescence" %in% optLayers) {
    p <- addStageLayers.ggplot(p, historical, stage_dat, 
                               crop$senescence, 
                               color = colors$senescence) +
      annotate("text", x = last_day, y = ystage_label, 
               label = "Senescence", hjust = 1,
               color = colors$senescence)
    ystage_label <- ystage_label + 100
  }
  if("layerFilling" %in% optLayers) {
    p <- addStageLayers.ggplot(p, historical, stage_dat, 
                               crop$filling, 
                               color = colors$filling) +
      annotate("text", x = last_day, y = ystage_label, 
               label = "Filling", hjust = 1,
               color = colors$filling)
    ystage_label <- ystage_label + 100
  }
  if("layerFlowering" %in% optLayers) {
    p <- addStageLayers.ggplot(p, historical, stage_dat, 
                               crop$flowering, 
                               color = colors$flowering) +
      annotate("text", x = last_day, y = ystage_label, 
               label = "Flowering", hjust = 1,
               color = colors$flowering)
    ystage_label <- ystage_label + 100
  }
  if("layerEmergence" %in% optLayers) {
    stage_date <- historical[cgdd_max>=crop$emergence]$date[1]
    p <- addStageLayers.ggplot(p, historical, stage_dat, 
                               crop$emergence, ystage_label,
                               color = colors) +
      annotate("text", x = stage_date, y = ystage_label, 
               label = "Emergence", 
               hjust = 0, vjust = -2, size = 4.5,
               color = 'gray10')
    #ystage_label <- ystage_label + 100
  }
  
  # Plot cfs ribbon
  if("layerCFS" %in% optLayers) {
    p <- p + 
      geom_ribbon(data = cfs, 
                  aes(ymin=cgdd_min, ymax=cgdd_max, x=date, group=model), 
                  alpha = 0.15,
                  fill=colors$cfs) + 
      geom_line(data = cfs, 
                aes(x = date, y = cgdd_mean, group=model), 
                linetype = 4,
                size = 1,
                color = colors$cfs) +
      annotate("text", x = first_day, y = ymax,
               label = "Climate Forecast", color = colors$cfs, 
               hjust = 0)
    ymax <- ymax - 100
  }
  # Plot nmme ribbon
  if("layerNMME" %in% optLayers) {
    p <- p + 
      geom_ribbon(data = nmme, 
                  aes(ymin=cgdd_min, ymax=cgdd_max, x=date),  
                  alpha = 0.15,
                  fill=colors$nmme) + 
      geom_line(data = nmme, 
                aes(x = date, y = cgdd_mean), 
                linetype = 4,
                size = 1,
                color = colors$nmme) +
      annotate("text", x = first_day, y = ymax,
               label = "NMME Forecast", color = colors$nmme, 
               hjust = 0)
    ymax <- ymax - 100
  }
  # Plot gefs ribbon
  if("layerGEFS" %in% optLayers) {
    p <- p + geom_ribbon(data = gefs, 
                         aes(ymin=cgdd_min, ymax=cgdd_max, x=date), 
                         alpha = .2,
                         fill="#C51B7D") + 
      geom_line(data = gefs, 
                aes(x = date, y = cgdd_mean), 
                linetype = 5,
                size = 1,
                color = "#C51B7D") + 
      annotate("text", x = first_day, y = ymax,
               label = "GEFS", color = "#C51B7D", 
               hjust = 0) 
    ymax <- ymax - 100
  } 
  
  p <- p + theme_bw() + setTheme(p)
  p
}



#' make a gdd plot of historical and future climate outlooks.
#' 
#' High level function to plot gdd climate outlook. Plots a distribution of 
#' historical and future cumulative GDD over the course of a year.
#' 
#' @export
#'
#' @param ... parameters passed to low level functions.
#' @param as_lines plot as lines instead of distribution.
#' @param engine rbokeh or ggplot2, determining which engine to use.
plot_gdd_outlook <- function(..., as_lines = FALSE, engine = c("plotly", "ggplot2")) {
  engine <- match.arg(engine)
  args <- list(...)
  if ("optLayers" %in% names(args) && length(args$optLayers != 0)) {
    if("all" %in% args$optLayers) {
      args$optLayers <- c(
        "layerFlowering",
        "layerFilling",
        "layerSenescence",
        "layerMaturity",
        "layerFutMinMax",
        "layerFut2sd",
        "layerFut1sd",
        "layerFutMean",
        "layerHistMinMax",
        "layerHist2sd",
        "layerHist1sd",
        "layerHistMean"
      )
    }
  }
  if(!"optLayers" %in% names(args)) args$optLayers <- NULL
  
  
  if (engine == "plotly") do.call(plot_gdd_outlook.plotly, args)
  else if (engine == "ggplot2") do.call(plot_gdd_outlook.ggplot, args) #(args)
}

#' Plot GDD with the plotly library
#' @export
#' @import plotly
#' @rdname plot_gdd
#' @return a GDD plot object made with plotly
plot_gdd_outlook.plotly <- function(crop, historical,
                                    fut2040, fut2060, fut2080, 
                                    stage_dates, 
                                    
                                    historical2,
                                    fut20402, fut20602, fut20802, 
                                    stage_dates2, 
                                    
                                    hist_last_frost_pct, hist_first_frost_pct,
                                    fut_last_frost_pct2040, fut_first_frost_pct2040,
                                    fut_last_frost_pct2060, fut_first_frost_pct2060,
                                    fut_last_frost_pct2080, fut_first_frost_pct2080,
                                    hist_med_len, fut2040_med_len, fut2060_med_len, fut2080_med_len,
                                    colors) {
  lblank <- list(color = toRGB("white", alpha=0)) #for the bottom range of ribbons, but invisible
  
  fl <- list(size = "18")
  fm <- list(size = "16")
  ft <- list(size = "14")
  
  yax1 <- list(anchor = "x", domain = c(0, .15), dtick = 4, zeroline = TRUE,
               title = "", showticklabels = FALSE)

  yax3 <- list(anchor = "x", domain = c(.15, .3), dtick = 4, zeroline = TRUE,
               side = 'left',
               title = "", showticklabels = FALSE)
  
  yax4 <- list(anchor = "x", domain = c(.3, .45), dtick = 4, zeroline = TRUE,
               side = 'left',
               title = "", showticklabels = FALSE)
  
  yax2 <- list(anchor = "x", domain = c(.45, 1), 
               title = "Cumulative GDD", titlefont = fl, tickfont = ft, zeroline = TRUE)
  
  #dates for plotly must be in milliseconds
  first_date <- as.numeric(min(historical$date)) * 24 * 60 *60 * 1000
  last_date <- as.numeric(max(historical$date)) * 24 * 60 *60 * 1000
  
  xax <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              hoverformat = "%b %e",
              range = c(first_date, last_date),
              tickfont = ft)
  leg <- list(x = 0,
              y = 1,
              xanchor = 'left',
              yanchor = 'left',
              orientation = 'h')
  
  p <- plot_ly() %>% layout(xaxis = xax, yaxis = yax1, yaxis2 = yax2, yaxis3 = yax3,  yaxis4 = yax4,
                            legend = leg,
                            
                            annotations = list(
                              list(xref = 'x', yref = 'y1',
                                   x = mean(hist_last_frost_pct$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   text = "Growing Season Start"
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(hist_first_frost_pct$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   text = "Growing Season End"
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   text = "Days"
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))),
                                   y = 3.5,
                                   showarrow = FALSE,
                                   text = hist_med_len
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))),
                                   y = 2.5,
                                   showarrow = FALSE,
                                   text = fut2040_med_len
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))),
                                   y = 1.5,
                                   showarrow = FALSE,
                                   text = fut2060_med_len
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))),
                                   y = 0.5,
                                   showarrow = FALSE,
                                   text = fut2080_med_len
                              ),
                              list(xref = 'x', yref = 'y1',
                                   x = first_date,
                                   y = 3.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "Historical"),
                              list(xref = 'x', yref = 'y1',
                                   x = first_date,
                                   y = 2.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2040"),
                              list(xref = 'x', yref = 'y1',
                                   x = first_date,
                                   y = 1.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2060"),
                              list(xref = 'x', yref = 'y1',
                                   x = first_date,
                                   y = 0.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2080"),
                              list(xref = 'x', yref = 'y3',
                                   x = first_date,
                                   y = 4.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   font = fm,
                                   text = "Shifted"
                              ),
                              list(xref = 'x', yref = 'y3',
                                   x = min(stage_dates[first_emergence_day == 1]$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   xanchor = "left",
                                   text = "Emergence Dates"),
                              list(xref = 'x', yref = 'y3',
                                   x = min(stage_dates[first_maturity_day == 1]$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   text = "Maturity Dates",
                                   xanchor = "center"),
                              
                              list(xref = 'x', yref = 'y3',
                                   x = first_date,
                                   y = 3.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "Historical"),
                              list(xref = 'x', yref = 'y3',
                                   x = first_date,
                                   y = 2.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2040"),
                              list(xref = 'x', yref = 'y3',
                                   x = first_date,
                                   y = 1.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2060"),
                              list(xref = 'x', yref = 'y3',
                                   x = first_date,
                                   y = 0.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2080"), 
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 4.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = "Post Maturity GDD"
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = first_date,
                                   y = 4.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   font = fm,
                                   text = "Default"
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = min(stage_dates[first_emergence_day == 1]$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   xanchor = "left",
                                   text = "Emergence Dates"),
                              list(xref = 'x', yref = 'y4',
                                   x = min(stage_dates[first_maturity_day == 1]$date),
                                   y = 4.5,
                                   showarrow = FALSE,
                                   text = "Maturity Dates",
                                   xanchor = "center"),
                              
                              list(xref = 'x', yref = 'y4',
                                   x = first_date,
                                   y = 3.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "Historical"),
                              list(xref = 'x', yref = 'y4',
                                   x = first_date,
                                   y = 2.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2040"),
                              list(xref = 'x', yref = 'y4',
                                   x = first_date,
                                   y = 1.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2060"),
                              list(xref = 'x', yref = 'y4',
                                   x = first_date,
                                   y = 0.5,
                                   showarrow = FALSE,
                                   xanchor = "right",
                                   text = "2080"),
                              list(xref = 'x', yref = 'y4',
                                   x = last_date,
                                   y = 3.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(historical[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - historical[date == paste0("", mean(stage_dates[year <= 2006 & first_maturity_day == 1]$date))]$cgdd_mean, 2) 
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = last_date,
                                   y = 2.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut2040[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut2040[date == paste0("", mean(stage_dates[year > 2025 & year <= 2055 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = last_date,
                                   y = 1.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut2060[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut2060[date == paste0("", mean(stage_dates[year > 2045 & year <= 2075 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = last_date,
                                   y = 0.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut2080[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut2080[date == paste0("", mean(stage_dates[year > 2065 & year <= 2095 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 3.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(historical2[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - historical2[date == paste0("", mean(stage_dates2[year <= 2006 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 2.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut20402[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut20402[date == paste0("", mean(stage_dates2[year > 2025 & year <= 2055 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 1.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut20602[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut20602[date == paste0("", mean(stage_dates2[year > 2045 & year <= 2075 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y3',
                                   x = last_date,
                                   y = 0.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = round(fut20802[date == paste0("", mean(hist_first_frost_pct$date))]$cgdd_mean - fut20802[date == paste0("", mean(stage_dates2[year > 2065 & year <= 2095 & first_maturity_day == 1]$date))]$cgdd_mean, 2)
                              ),
                              list(xref = 'x', yref = 'y4',
                                   x = last_date,
                                   y = 4.5,
                                   xanchor = "right",
                                   showarrow = FALSE,
                                   text = "Post Maturity GDD"
                              )
                            ))

  ### HISTORICAL, FUTURE, and FORECAST DATA
  ## FUTURE 2080
  # min/max
  lf2080 <- list(color = toRGB(colors$fut2080, alpha=0.3))
  p <- add_trace(p, data = fut2080, x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank,
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_max, 
              xaxis = "x1", yaxis = "y2",
              visible = "legendonly",
              mode="lines", fill = "tonexty", line = lf2080, 
              name = "2080 0-100%", hoverinfo = "none")
  
  # 10-90%
  p <- add_trace(p, data = fut2080, x = date, y = cgdd_10pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_90pct, 
              xaxis = "x1", yaxis = "y2",
              visible = "legendonly",
              mode="lines", fill = "tonexty", line = lf2080, 
              name = "2080 10-90%", hoverinfo = "none")
  
  # 25-75%
  p <- add_trace(p, data = fut2080, x = date, y = cgdd_25pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_75pct, 
              xaxis = "x1", yaxis = "y2",
              visible = "legendonly",
              mode="lines", fill = "tonexty", line = lf2080, 
              name = "2080 25-75%", hoverinfo = "none")
  
  # Mean
  lf2080_mean <- list(color = toRGB(colors$fut2080, alpha=0.9))
  p <- add_trace(p, data = fut2080, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 line = lf2080_mean, 
                 showlegend = FALSE,
                 text = paste("2080 Mean:", round(fut2080$cgdd_mean,0)), 
                 name = "2080 Mean", 
                 hoverinfo = "x+text")
  
  
  ## FUTURE 2060
  #min/max
  lf2060 <- list(color = toRGB(colors$fut2060, alpha=0.3))
  p <- add_trace(p, data = fut2060, x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank,
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_max, 
                 xaxis = "x1", yaxis = "y2",
                 visible = "legendonly",
                 mode="lines", fill = "tonexty", line = lf2060, 
                 name = "2060 0-100%", hoverinfo = "none")
  
  # 10-90%
  lf2sd <- list(color = toRGB(colors$fut_2sd, alpha=0.3))
  p <- add_trace(p, data = fut2060, x = date, y = cgdd_10pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_90pct, 
                 xaxis = "x1", yaxis = "y2",
                 visible = "legendonly",
                 mode="lines", fill = "tonexty", line = lf2060, 
                 name = "2060 10-90%", hoverinfo = "none")
  
  # 25-75%
  lf1sd <- list(color = toRGB(colors$fut_2sd, alpha=0.3))
  p <- add_trace(p, data = fut2060, x = date, y = cgdd_25pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
       add_trace(x = date, y = cgdd_75pct, 
                 xaxis = "x1", yaxis = "y2",
                 visible = "legendonly",
                 mode="lines", fill = "tonexty", line = lf2060, 
                 name = "2060 25-75%", hoverinfo = "none")
  
  # Mean
  lf2060_mean <- list(color = toRGB(colors$fut2060, alpha=0.9))
  p <- add_trace(p, data = fut2060, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 line = lf2060_mean, 
                 showlegend = FALSE,
                 text = paste("2060 Mean:", round(fut2060$cgdd_mean,0)), 
                 name = "2060 Mean", 
                 hoverinfo = "x+text")
  
  
  ### FUTURE 2040
  # min/max
  lf2040 <- list(color = toRGB(colors$fut2040, alpha=0.3))
  p <- add_trace(p, data = fut2040, x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank,
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_max, 
              xaxis = "x1", yaxis = "y2",
              visible = "legendonly",
              mode="lines", fill = "tonexty", line = lf2040, 
              name = "2040 0-100%", hoverinfo = "none")
  
  # 10-90%
  p <- add_trace(p, data = fut2040, x = date, y = cgdd_10pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_90pct, 
              xaxis = "x1", yaxis = "y2",
              visible = "legendonly",
              mode="lines", fill = "tonexty", line = lf2040, 
              name = "2040 10-90%", hoverinfo = "none")
  
  # 25-75%
  p <- add_trace(p, data = fut2040, x = date, y = cgdd_25pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 visible = "legendonly",
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_75pct, 
              xaxis = "x1", yaxis = "y2",
              mode="lines", fill = "tonexty", line = lf2040,
              visible = "legendonly",
              name = "2040 25-75%", hoverinfo = "none")
  
  # Mean
  lf2040_mean <- list(color = toRGB(colors$fut2040, alpha=0.9))
  p <- add_trace(p, data = fut2040, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 line = lf2040_mean, 
                 showlegend = FALSE,
                 text = paste("2040 Mean:", round(fut2040$cgdd_mean,0)), 
                 name = "2040 Mean", 
                 hoverinfo = "x+text")
  
  
  # Historical min/max
  lhist <- list(color = toRGB(colors$hist, alpha=0.3))
  p <- add_trace(p, data = historical, x = date, y = cgdd_min, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank,
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_max, 
              xaxis = "x1", yaxis = "y2",
              mode="lines", fill = "tonexty", line = lhist, 
              name = "Historical 0-100%", hoverinfo = "none")
  
  # Historical 2 standard deviations 
  lh2sd <- list(color = toRGB(colors$hist, alpha=0.3))
  p <- add_trace(p, data = historical, x = date, y = cgdd_10pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_90pct, 
              xaxis = "x1", yaxis = "y2",
              mode="lines", fill = "tonexty", line = lhist, 
              name = "Historical 10-90%", hoverinfo = "none")
  
  # Historical 1 standard deviation
  lh1sd <- list(color = toRGB(colors$hist, alpha=0.3))
  p <- add_trace(p, data = historical, x = date, y = cgdd_25pct, 
                 xaxis = "x1", yaxis = "y2",
                 mode="lines", line = lblank, 
                 showlegend = FALSE, hoverinfo = "none") %>%
    add_trace(x = date, y = cgdd_75pct, 
              xaxis = "x1", yaxis = "y2",
              mode="lines", fill = "tonexty", line = lhist, 
              name = "Historical 25-75%", hoverinfo = "none")
  
  # Historical mean
  lhist_mean <- list(color = toRGB(colors$hist, alpha=0.9))
  p <- add_trace(p, data = historical, x = date, y = cgdd_mean, 
                 xaxis = "x1", yaxis = "y2",
                 line = lhist_mean, 
                 showlegend = FALSE,
                 text = paste("Historical Mean:", round(historical$cgdd_mean,0)), 
                 name = "Historical Mean", 
                 hoverinfo = "x+text")

  p <- add_trace(p, data = hist_last_frost_pct,  x = date, hoverinfo="x+text", showlegend = FALSE,
            type = "box", orientation = "h", line=list(color=palette3$hist), 
            name = "Historical Growing Season Start", y0 = 3.5, xaxis = "x1", yaxis = "y1") %>%
    add_trace(data = hist_first_frost_pct, x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h", line=list(color=palette3$hist), 
              name = "Historical Growing Season End", y0 = 3.5, xaxis = "x1", yaxis = "y1") %>%
    
    add_trace(data = fut_last_frost_pct2040,  x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h", line=list(color=palette3$fut2040), 
              name = "2040  Growing Season Start", y0 = 2.5, xaxis = "x1", yaxis = "y1") %>%
    add_trace(data = fut_first_frost_pct2040, x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h", line=list(color=palette3$fut2040),  
              name = "2040 Growing Season End", y0 = 2.5, xaxis = "x1", yaxis = "y1") %>%
    
    add_trace(data = fut_last_frost_pct2060,  x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h", line=list(color=palette3$fut2060), 
              name = "2060  Growing Season Start",y0 = 1.5, xaxis = "x1", yaxis = "y1") %>%
    add_trace(data = fut_first_frost_pct2060, x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h", line=list(color=palette3$fut2060), 
              name = "2060 Growing Season End", y0 = 1.5, xaxis = "x1", yaxis = "y1") %>%
    
    add_trace(data = fut_last_frost_pct2080,  x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h",line=list(color=palette3$fut2080),  
              name = "2080  Growing Season Start", y0 = 0.5, xaxis = "x1", yaxis = "y1") %>%
    add_trace(data = fut_first_frost_pct2080, x = date, hoverinfo="x+text", showlegend = FALSE,
              type = "box", orientation = "h",line=list(color=palette3$fut2080), 
              name = "2080 Growing Season End", y0 = 0.5, xaxis = "x1", yaxis = "y1")
  

  # Stage Bars
  p <- add_trace(p, data = stage_dates[year > 2065 & year <= 2095 & first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2080),
                 showlegend = FALSE, name = "Future 2080", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 0.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year > 2065 & year <= 2095 & first_maturity_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2080),
                 showlegend = FALSE, name = "Future 2080", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 0.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year > 2045 & year <= 2075 & first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2060),
                 showlegend = FALSE, name = "Future 2060", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 1.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year > 2045 & year <= 2075 & first_maturity_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2060),
                 showlegend = FALSE, name = "Future 2060", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 1.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year > 2025 & year <= 2055 & first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2040),
                 showlegend = FALSE, name = "Future 2040", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 2.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year > 2025 & year <= 2055 & first_maturity_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2040),
                 showlegend = FALSE, name = "Future 2040", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 2.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year <= 2006 & first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$hist),
                 showlegend = FALSE, name = "Historical", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 3.5, xaxis = "x1", yaxis = "y4") %>%
       add_trace(data = stage_dates[year <= 2006 & first_maturity_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$hist),
                 showlegend = FALSE, name = "Historical", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 3.5, xaxis = "x1", yaxis = "y4")
  
  
  # Stage Bars 2
  p <- add_trace(p, data = stage_dates2[year > 2065 & year <= 2095 & first_emergence_day == 1], x = date,
                 type = "box", orientation = "h", line = list(color = colors$fut2080),
                 showlegend = FALSE, name = "Future 2080", hoverinfo="x+text",
                 text = paste("Mean Start Date:", mean(date)), 
                 y0 = 0.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year > 2065 & year <= 2095 & first_maturity_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$fut2080),
              showlegend = FALSE, name = "Future 2080", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 0.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year > 2045 & year <= 2075 & first_emergence_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$fut2060),
              showlegend = FALSE, name = "Future 2060", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 1.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year > 2045 & year <= 2075 & first_maturity_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$fut2060),
              showlegend = FALSE, name = "Future 2060", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 1.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year > 2025 & year <= 2055 & first_emergence_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$fut2040),
              showlegend = FALSE, name = "Future 2040", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 2.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year > 2025 & year <= 2055 & first_maturity_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$fut2040),
              showlegend = FALSE, name = "Future 2040", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 2.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year <= 2006 & first_emergence_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$hist),
              showlegend = FALSE, name = "Historical", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 3.5, xaxis = "x1", yaxis = "y3") %>%
    add_trace(data = stage_dates2[year <= 2006 & first_maturity_day == 1], x = date,
              type = "box", orientation = "h", line = list(color = colors$hist),
              showlegend = FALSE, name = "Historical", hoverinfo="x+text",
              text = paste("Mean Start Date:", mean(date)), 
              y0 = 3.5, xaxis = "x1", yaxis = "y3")

  #return the plot
  p
}

