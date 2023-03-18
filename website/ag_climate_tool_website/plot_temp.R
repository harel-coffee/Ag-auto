#' @export 
c_to_f <- function(temp){
  (temp*9/5) + 32
}

plot_temp_forecast <- function(hist, pd, gefs, nmme, 
                               hist_last_frost_pct, hist_first_frost_pct, 
                               hist_med_len) {
  
  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s2 <- list(title = "", titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             hoverformat = "%b %e",
             tickfont = ft) 
  
  s <- list(anchor = 'x', domain = c(0, 0.10), dtick = 5, 
            title = "",
            titlefont = fl, tickfont = ft)
  
  s1 <- list(anchor = 'x', domain = c(0.15, 1), dtick = 5, 
             title = "Temperature (˚C)",
             titlefont = fl, tickfont = ft)

  p <- plot_ly()  %>%
    
    layout(yaxis = s, yaxis2 = s1, xaxis = s2,
           annotations = list(
             list(xref = 'x', yref = 'y1',
                  x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))) ,
                  y = 0,
                  showarrow = FALSE,
                  text = hist_med_len
             ),
             list(xref = 'x', yref = 'y1',
                  x = mean(hist_last_frost_pct$date),
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
             list(xref = 'x', yref = 'y1',
                  x = mean(c(mean(hist_last_frost_pct$date),  mean(hist_first_frost_pct$date))) ,
                  y = 1,
                  showarrow = FALSE,
                  text = "Len"
             )
           ))  %>%
    
    add_trace(data = hist, x = date, y = temp_max_10pct, 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist, width=0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_max_90pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist, width = 0), fill = "tonexty", 
              name="10-90% of Historical Maximum", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_max_25pct, hoverinfo = "x+y+text", 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist, width=0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_max_75pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist, width=0), fill = "tonexty", 
              name = "25-75% of Historical Maximum", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_max_mean, 
              mode="lines",  line=list(color=palette3$hist),  
              hoverinfo = "x+y+text", text = paste("Mean Historical Maximum:", round(temp_max_mean,1)),
              name = "Mean of Historical Maximum", xaxis = "x1", yaxis = "y2") %>%
    
    add_trace(data = hist, x = date, y = temp_mean_10pct, 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist2, width = 0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_mean_90pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist2, width = 0), fill = "tonexty", 
              name = "10-90% of Historical Mean", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_mean_25pct, hoverinfo = "x+y+text", 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist2, width = 0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_mean_75pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist2, width = 0), fill = "tonexty", 
              name = "25-75%% of Historical Mean", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_mean_mean, 
              mode="lines", line=list(color=palette3$hist2), 
              hoverinfo = "x+y+text", text = paste("Mean Historical Mean:", round(temp_mean_mean,1)),
              name = "Mean of Historical Mean", xaxis = "x1", yaxis = "y2") %>%
    
    add_trace(data = hist, x = date, y = temp_min_10pct, 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist3, width = 0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_min_90pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist3, width = 0), fill = "tonexty", 
              name = "10-90% of Historical Mininum", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_min_25pct, hoverinfo = "x+y+text", 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$hist3, width = 0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_min_75pct, hoverinfo = "x+y+text", 
              mode="lines", line=list(color=palette3$hist3, width  = 0), fill = "tonexty", 
              name = "25-75%% of Historical Mininum", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = hist, x = date, y = temp_min_mean, 
              mode="lines", line=list(color=palette3$hist3), 
              hoverinfo = "x+y+text", text = paste("Mean Historical Minimum:", round(temp_min_mean,1)),
              name = "Mean of Historical Mininum", xaxis = "x1", yaxis = "y2") %>%
    
    add_trace(data = pd, x = date, y = temp_min, 
              mode="lines", line=list(color=palette3$present),
              hoverinfo = "x+y+text", text = paste("Year-to-Date Minimum:", round(temp_min,1)),
              name = "Year-to-Date Minimum", visible = "legendonly", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = pd, x = date, y = temp_max, 
              mode="lines", line=list(color=palette3$present),  
              hoverinfo = "x+y+text", text = paste("Year-to-Date Maximum:", round(temp_max,1)),
              name = "Year-to-Date Maximum", visible = "legendonly", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = pd, x = date, y = temperature,
              mode="lines", line=list(color=palette3$present), 
              hoverinfo = "x+y+text", text = paste("Year-to-Date Mean:", round(temperature,1)),
              name = "Year-to-Date Mean", visible = "legendonly",xaxis = "x1", yaxis = "y2") %>%
    
    ######## Range from max - min not showing up
    add_trace(data = gefs, x = date, y = temp_min, 
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$gefs, width = 0), xaxis = "x1", yaxis = "y2") %>%
    
    
    add_trace(data = gefs, x = date, y = temp_max,  hoverinfo = "x+y+text",
              mode="lines", line=list(color=palette3$gefs, width = 0), 
              fill = "tonexty",visible = "legendonly", 
              name = "Short-Term Forecast (GEFS) Min - Max", xaxis = "x1", yaxis = "y2") %>%
    ########   
    
    
    add_trace(data = gefs, x = date, y = temp_mean, 
              mode="lines",  line=list(color=palette3$gefs), visible = "legendonly",
              hoverinfo = "x+y+text", text = paste("Short-Term Forecast Mean:", round(temp_mean, 1)),
              name = "Short-Term Forecast (GEFS) Mean", xaxis = "x1", yaxis = "y2") %>%
    
    
    
    
    add_trace(data = nmme, x = date, y = temp_min,
              mode="lines", showlegend = FALSE, hoverinfo = "x+y+text", 
              line=list(color=palette3$nmme, width = 0), xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = nmme, x = date, y = temp_max, hoverinfo='text',
              mode="lines", line=list(color=palette3$nmme, width = 0), fill = "tonexty",visible = "legendonly", 
              name = "Long-Term Forecast (NMME) Min - Max", xaxis = "x1", yaxis = "y2") %>%
    add_trace(data = nmme, x = date, y = temp_mean, 
              mode="lines", line=list(color=palette3$nmme), visible = "legendonly", 
              hoverinfo="text", text = paste("Long-Term Forecast Mean:", round(temp_mean, 1)),
              name = "Long-Term Forecast (NMME) Mean", xaxis = "x1", yaxis = "y2") %>%
    
    add_trace(data = hist_last_frost_pct, x = date, 
              type = "box", orientation = "h", line = list(color = palette3$hist), 
              showlegend = FALSE, hoverinfo="text",
              text = paste("Mean Start Date:", mean(date)), 
              name = "Growing Season Start", y0 = 0, xaxis = "x1", yaxis = "y1")  %>%
    
    add_trace(data = hist_first_frost_pct, x = date, 
              type = "box", orientation = "h", line = list(color = palette3$hist), 
              showlegend = FALSE, hoverinfo="text",
              text = paste("Mean Start Date:", mean(date)), 
              name = "Growing Season End", y0 = 0, xaxis = "x1", yaxis = "y1")
  
  
}

#' @export 
plot_temp_outlook1 <- function(hist, fut2040, fut2060, fut2080)
{
  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s <- list(titlefont = fl, tickfont = ft, showgrid = TRUE, title = "Temperature (°F)")
  
  s2 <- list(titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%
    
    layout(xaxis = s2, yaxis = s, boxmode='group', legend=list(x=0, y=200))  %>%
    add_trace(data = hist, x = date, y = c_to_f(temp_max), type="box", line=list(color=palette3$hist), name = "Historical", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2040, x = date, y = c_to_f(temp_max), type="box", line=list(color=palette3$fut2040), name = "2040", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2060, x = date, y = c_to_f(temp_max), type="box", line=list(color=palette3$fut2060), name = "2060", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2080, x = date, y = c_to_f(temp_max), type="box", line=list(color=palette3$fut2080), name = "2080", hoverinfo = "x+y+text")
    
}

#' @export 
plot_temp_outlook2 <- function(hist, fut2040, fut2060, fut2080)
{
  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s <- list(titlefont = fl, tickfont = ft, showgrid = TRUE, title = "Temperature (°F)")
  
  s2 <- list(titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             tickfont = ft, showgrid = TRUE)
  
  p <- plot_ly() %>%
    layout(xaxis = s2, yaxis = s, boxmode='group', legend=list(x=0, y=200))  %>%
    add_trace(data = hist, x = date, y = c_to_f(temp_mean), type="box", line=list(color=palette3$hist), name = "Historical", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2040, x = date, y = c_to_f(temp_mean), type="box", line=list(color=palette3$fut2040), name = "2040", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2060, x = date, y = c_to_f(temp_mean), type="box", line=list(color=palette3$fut2060), name = "2060", hoverinfo = "x+y+text") %>%
    add_trace(data = fut2080, x = date, y = c_to_f(temp_mean), type="box", line=list(color=palette3$fut2080), name = "2080", hoverinfo = "x+y+text")
  
}

#' @export 
plot_temp_outlook3 <- function(hist, fut2040, fut2060, fut2080)
{

  fl <- list(size = "18")
  ft <- list(size = "14")
  
  s <- list(titlefont = fl, tickfont = ft, showgrid = TRUE, title = "Temperature (°F)")
  
  s2 <- list(titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             tickfont = ft, showgrid = TRUE)

  p <- plot_ly() %>%
    
      layout(xaxis = s2, yaxis = s, boxmode='group', legend=list(x=0, y=200))  %>%
      
      add_trace(data = hist, x = date, y = c_to_f(temp_min), type="box", line=list(color=palette3$hist),  xaxis = "x1", name = "Historical", hoverinfo = "x+y+text") %>%
      add_trace(data = fut2040, x = date, y = c_to_f(temp_min), type="box", line=list(color=palette3$fut2040), xaxis = "x1", name = "2040", hoverinfo = "x+y+text") %>%
      add_trace(data = fut2060, x = date, y = c_to_f(temp_min), type="box", line=list(color=palette3$fut2060), xaxis = "x1",  name = "2060", hoverinfo = "x+y+text") %>%
      add_trace(data = fut2080, x = date, y = c_to_f(temp_min), type="box", line=list(color=palette3$fut2080), xaxis = "x1", name = "2080", hoverinfo = "x+y+text")
}


#' @export 
plot_temp_heat_risk <- function(hist, fut3_40, fut3_60 ,fut3_80) 
{
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Days Above Threshold", 
            titlefont = fl, tickfont = ft)
  s2 <- list(title = "", titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             tickfont = ft)
  
  p <- plot_ly() %>%
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%   
    add_trace(data = hist, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$hist),
              name = "Historical", hovertext="text") %>%  
    add_trace(data = fut3_40, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$fut2040), 
              name = "2040", hovertext="text") %>%   
    add_trace(data = fut3_60, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$fut2060), 
              name = "2060", hovertext="text") %>%   
    add_trace(data = fut3_80, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$fut2080), 
              name = "2080", hovertext="text")
  
}

#' @export 
plot_temp_frost_risk <- function(hist, fut3_40, fut3_60 ,fut3_80) 
{
  fl <- list(size = "18")
  ft <- list(size = "14")
  s <- list(title = "Days Below Freezing", 
            titlefont = fl, tickfont = ft)
  s2 <- list(title = "", titlefont = fl,
             nticks = 12,
             tickformat = "%b",
             tickfont = ft)
  
  p <- plot_ly() %>%
    layout(yaxis = s, xaxis = s2, boxmode='group',  legend=list(x=0, y=200)) %>%   
    add_trace(data = hist, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$hist), 
              name = "Historical", hovertext="text") %>% 
    add_trace(data = fut3_40, x = date, y = risk_count, 
              type ="box",line=list(color=palette3$fut2040), 
              name = "2040", hovertext="text") %>%   
    add_trace(data = fut3_60, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$fut2060), 
              name = "2060", hovertext="text") %>%   
    add_trace(data = fut3_80, x = date, y = risk_count, 
              type ="box", line=list(color=palette3$fut2080), 
              name = "2080", hovertext="text")
}




