#' @export
mm_to_in <- function(precip){
  precip/25.4
}

plot_precip_forecast_snow <- function(hist, snow, gefs, nmme, ht) {
  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s <- list(title = "Snow Precipitation (in)", 
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  
  s2 <- list(titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%  
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(snow), 
              type = "box", line = list(color=palette3$hist),
              text = paste("Historical Mean:", round(mean(hist$snow),0)),
              hoverinfo = "x+y+text",
              name = "Historical") %>%
    add_trace(data = snow, x = date, y = mm_to_in(cum_snow), 
              mode="lines",  line=list(color=palette3$present, width=0), fill = "tozeroy", 
              text = paste("Year-to-Date:", round(snow$cum_snow,0)),
              hoverinfo = "x+text",
              name = "Present to Date") %>%
    add_trace(data = gefs, x = date, y = mm_to_in(max_snow), mode="lines", line=list(color=palette3$gefs), 
              text = paste("Short-Term Forecast:", round(gefs$max_snow,0)),
              hoverinfo = "x+text",
              name = "Short-Term Forecast (GEFS)") %>%
    add_trace(data = nmme, x = date, y = mm_to_in(max_snow), mode="lines", line=list(color=palette3$nmme), 
              text = paste("Long-Term Forecast:", round(nmme$max_snow,0)),
              hoverinfo = "x+text",
              name = "Long-Term Forecast (NMME)")
}

#' @export
plot_precip_forecast_rain <- function(hist, rain, gefs, nmme, ht) 
{ 
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Rain Precipitation (in)",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)

  p <- plot_ly()  %>%
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(rain), 
              type="box",line=list(color=palette3$hist), 
              text = paste("Historical Mean:", round(mean(hist$rain),0)),
              hoverinfo = "x+y+text",
              name = "Historical") %>%
    add_trace(data = rain, x = date, y = mm_to_in(cum_rain), 
              mode="lines", line=list(color=palette3$present, width=0), fill = "tozeroy", 
              text = paste("Year-to-Date:", round(rain$cum_rain,0)),
              hoverinfo = "x+text",
              name = "Year-to-Date") %>%
    add_trace(data = gefs, x = date, y = mm_to_in(max_rain), 
              mode="lines", line=list(color=palette3$gefs), 
              text = paste("Short-Term Forecast:", round(gefs$max_rain,0)),
              hoverinfo = "x+text",
              name = "Short-Term Forecast (GEFS)") %>%
    add_trace(data = nmme, x = date, y = mm_to_in(max_rain), 
              mode="lines", line=list(color=palette3$nmme), 
              text = paste("Long-Term Forecast:", round(gefs$max_rain,0)),
              hoverinfo = "x+text",
              name = "Long-Term Forecast (NMME)")
}

#' @export
plot_precip_forecast_total <- function(hist, pd, gefs, nmme) 
{
  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s <- list(title = "Total Precipitation (in)",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  p <- plot_ly() %>%
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(rain + snow), 
              type="box", line=list(color=palette3$hist), 
              text = paste("Historical Mean:", mean(hist$rain + hist$snow)),
              hoverinfo = "x+y+text",
              name = "Historical")  %>%  
    add_trace(data = pd,x = date, y = mm_to_in(cum_rain + cum_snow), 
              mode="lines", line=list(color=palette3$present, width=0), fill = "tozeroy", 
              text = paste("Year-to-Date:", round(pd$cum_rain + pd$cum_snow,0)),
              hoverinfo = "x+text",
              name = "Year-to-Date") %>%
    add_trace(data = gefs, x = date, y = mm_to_in(max_rain + max_snow), 
              mode="lines", line=list(color=palette3$gefs), 
              text = paste("Short-Term Forecast:", round(gefs$max_rain + gefs$max_snow,0)),
              hoverinfo = "x+text",
              name = "Short-Term Forecast (GEFS)") %>%
    add_trace(data = nmme, x = date, y = mm_to_in(max_rain + max_snow), 
              mode="lines", line=list(color=palette3$nmme), 
              text = paste("Long-Term Forecast:", round(nmme$max_rain + nmme$max_snow,0)),
              hoverinfo = "x+text",
              name = "Long-Term Forecast (NMME)")
}

#' @export
plot_precip_outlook_snow <- function(hist, fut3_40, fut3_60, fut3_80, ht) 
{ 
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Snow Precipitation (in)",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%  
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(snow), 
              type="box", line=list(color=palette3$hist),
              text = paste("Historical Mean", round(mean(hist$snow),0)),
              hoverinfo = "x+y+text",
              name = "Historical") %>%
    add_trace(data = fut3_40, x = date, y = mm_to_in(snow), 
              type="box", line=list(color=palette3$fut2040), 
              text = paste("2040 Mean:", round(mean(fut3_40$snow),0)),
              hoverinfo = "x+y+text",
              name = "2040") %>%
    add_trace(data = fut3_60, x = date, y = mm_to_in(snow), 
              type="box", line=list(color=palette3$fut2060), 
              text = paste("2060 Mean:", round(mean(fut3_60$snow),0)),
              hoverinfo = "x+y+text",
              name = "2060") %>%
    add_trace(data = fut3_80, x = date, y = mm_to_in(snow), 
              type="box", line=list(color=palette3$fut2080), 
              text = paste("2080 Mean:", round(mean(fut3_80$snow),0)),
              hoverinfo = "x+y+text",
              name = "2080")
}

#' @export
plot_precip_outlook_rain <- function(hist, fut3_40, fut3_60, fut3_80, ht)  
{ 
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Rain Precipitation (in)",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%  
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(rain), 
              type="box", line=list(color=palette3$hist), 
              text = paste("Historical Mean:", round(mean(hist$rain),0)),
              hoverinfo = "x+y+text",
              name = "Historical") %>%
    add_trace(data = fut3_40, x = date, y = mm_to_in(rain), 
              type="box", line=list(color=palette3$fut2040), 
              text = paste("2040 Mean:", round(mean(fut3_40$rain),0)),
              hoverinfo = "x+y+text",
              name = "2040")  %>%
    add_trace(data = fut3_60, x = date, y = mm_to_in(rain), 
              type="box", line=list(color=palette3$fut2060), 
              text = paste("2060 Mean:", round(mean(fut3_60$rain),0)),
              hoverinfo = "x+y+text",
              name = "2060")  %>%
    add_trace(data = fut3_80, x = date, y = mm_to_in(rain), 
              type="box", line=list(color=palette3$fut2080), 
              text = paste("2080 Mean:", round(mean(fut3_80$rain),0)),
              hoverinfo = "x+y+text",
              name = "2080")
}

#' @export
plot_precip_outlook_total <- function(hist, fut3_40, fut3_60, fut3_80) 
{ 
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Total Precipitation (in)",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist, x = date, y = mm_to_in(hist$rain+hist$snow), 
              type="box",line=list(color=palette3$hist), 
              text = paste("Historical Mean:", round(mean(hist$rain+hist$snow),0)),
              hoverinfo = "x+y+text",
              name = "Historical") %>%
    add_trace(data = fut3_40, x = date, y = mm_to_in(fut3_40$rain+fut3_40$snow),
              type="box", line=list(color=palette3$fut2040), 
              text = paste("2040 Mean:", round(mean(fut3_40$rain+fut3_40$snow),0)),
              hoverinfo = "x+y+text",
              name = "2040") %>%
    add_trace(data = fut3_60, x = date, y = mm_to_in(fut3_60$rain+fut3_60$snow), 
              type="box", line=list(color=palette3$fut2060), 
              text = paste("2060 Mean:", round(mean(fut3_60$rain+fut3_60$snow),0)),
              hoverinfo = "x+y+text",
              name = "2060") %>%
    add_trace(data = fut3_80, x = date, y = mm_to_in(fut3_80$rain+fut3_80$snow), 
              type="box", line=list(color=palette3$fut2080), 
              text = paste("2080 Mean:", round(mean(fut3_80$rain+fut3_80$snow),0)),
              hoverinfo = "x+y+text",
              name = "2080")
}

#' @export
plot_precip_outlook_frequency <- function(fut3_40, fut3_60, fut3_80, hist3) {
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Days Above Precipitation Threshold",
            titlefont = fl, tickfont = ft, showgrid = TRUE)
  s2 <- list(title = "", titlefont = fl,
              nticks = 12,
              tickformat = "%b",
              tickfont = ft, showgrid = TRUE)
  
  hi_hist <- boxplot.stats(hist3$risk_count)$stats
  hi_2040 <- boxplot.stats(fut3_40$risk_count)$stats
  hist_text <- paste("Historical Mean:", hi_hist[3])
  
  p <- plot_ly()  %>%  
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%  
    add_trace(data = hist3, x = date, y = risk_count, 
              type="box", line=list(color=palette3$hist), 
              text = hist_text,
              hoverinfo = "x+y+text",
              #text = paste("Historical Mean:", hi_hist[3]),
              name = "Historical")  %>%
    add_trace(data = fut3_40, x = date, y = risk_count, 
              type="box", line=list(color=palette3$fut2040), 
              text = paste("2040 Mean:", hi_2040[3]),
              hoverinfo = "x+y+text",
              name = "2040")  %>%   
    add_trace(data = fut3_60, x = date, y = risk_count, 
              type="box", line=list(color=palette3$fut2060), 
              text = paste("2060 Mean:", mean(fut3_60$risk_count)),
              hoverinfo = "x+y+text",
              name = "2060")  %>%  
    add_trace(data = fut3_80, x = date, y = risk_count, 
              type="box", line=list(color=palette3$fut2080), 
              text = paste("2080 Mean:", mean(fut3_80$risk_count)),
              hoverinfo = "x+y+text",
              name = "2080")
  p
}








































