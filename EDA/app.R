library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(knitr)
library(htmltools)
library(DT)

# Dataset to be explored. 
raw_df <- read.csv("C:\\PestList\\MyDocuments\\Statistic\\ST558\\ShinyApps\\FinalProject\\NHANES_age_prediction.csv")

# Pre-compute some variables to be used by app
# convert some variable to factor
raw_df$RIAGENDR <- factor(raw_df$RIAGENDR, levels = c(1, 2), labels = c("Male", "Female"))
raw_df$PAQ605 <- factor(raw_df$PAQ605, levels = c(1, 2, 7), labels = c("Yes", "No", "Refused"))
raw_df$DIQ010 <- factor(raw_df$DIQ010, levels = c(1, 2, 3), labels = c("Yes", "No", "Prediabetes"))

character <- sapply(names(raw_df), function(x) !is.numeric(raw_df[[x]]))
character_vars <- names(raw_df)[sapply(raw_df, function(x) !is.numeric(x))]
numeric <- setdiff(names(raw_df)[sapply(raw_df, is.numeric)], "SEQN")

df <- raw_df

# Define UI ----
ui <- navbarPage("Dataset Explorer Analysis",
                 tabPanel("Plot Type", 
                          fluidRow(
                            sidebarLayout(
                              sidebarPanel(
                                h4("Subset data by gender:"),
                                selectInput("RIAGENDR", "Gender", selected = "Male", choices = levels(as.factor(raw_df$RIAGENDR))),
                                br(),
                                selectInput("plotType", "Plot Type",
                                            choices = c("Scatter Plot", "Boxplot", "BarPlot", "Histogram"),
                                            selected = "Scatter Plot"),
                                # Only show this panel if the plot type is a scatter plot
                                conditionalPanel(
                                  condition = "input.plotType == 'Scatter Plot'",
                                  selectInput("scatx", "X-axis Variable", 
                                              choices = numeric, 
                                              selected = "RIDAGEYR"),
                                  selectInput("scaty", "Y-axis Variable", 
                                              choices = numeric, 
                                              selected = "RIDAGEYR")
                                ),
                                conditionalPanel(
                                  condition = "input.plotType == 'Boxplot'",
                                  
                                  selectInput("boxx", "X-axis Variable", 
                                              choices = character_vars, 
                                              selected = "age_group"),
                                  selectInput("boxy", "Y-axis Variable", 
                                              choices = numeric,  
                                              selected = "RIDAGEYR")
                                ),
                                conditionalPanel(
                                  condition = "input.plotType == 'BarPlot'",
                                  selectInput("barx", "X-axis Variable", 
                                              choices = character_vars, selected = "age_group"),
                                  selectInput("bary", "Y-axis Variable", 
                                              choices = character_vars, 
                                              selected = "age_group")
                                ),
                                conditionalPanel(
                                  condition = "input.plotType == 'Histogram'",
                                  selectInput("histx", "X-axis Variable", 
                                              choices = numeric,
                                              selected = "RIDAGEYR")
                                ),
                                # only allow non-numeric variables for color
                                selectInput("color", "Color", c("None", names(df)[character])),
                                p("Smoothing is only available when two numeric variables are selected."),
                                checkboxInput("smooth", "Smooth")
                              ),
                              
                              mainPanel(conditionalPanel(
                                condition = "input.plotType == 'Scatter Plot'",
                                plotOutput("plotType")
                              ),
                              conditionalPanel(
                                condition = "input.plotType == 'Boxplot'",
                                plotOutput("plot2")
                              ),
                              conditionalPanel(
                                condition = "input.plotType == 'BarPlot'",
                                plotOutput("plot3")
                              ),
                              conditionalPanel(
                                condition = "input.plotType == 'Histogram'",
                                plotOutput("plot4")
                              )
                              )
                            )
                          )
                 ),
                 
                 
                 tabPanel("Cor_Matrix", dataTableOutput("correlation")),
                 
                 tabPanel("Summary", 
                          h5("Numeric summary is for the full data set."),
                          radioButtons("summary", "Numeric Summary", 
                                       choices = c("Q1", "Q3", "Mean", "Median"), 
                                       selected = "Mean"),
                          mainPanel(
                            DTOutput("summary"))
                 ),
                 
                 tabPanel("Contigency Table",
                          h5("Contingency table is for the full data set."),
                          sidebarPanel(
                            h4("Character Summary"),
                            selectInput("var1", "Variable 1", character_vars),
                            selectInput("var2", "Variable 2", character_vars)
                          ),
                          
                          mainPanel(
                            dataTableOutput("table")
                          )
                 )
)


# Define server logic ----
server <- function(input, output, session) {
  #get data for only gender group specified
  getData <- reactive({
    genders <- input$RIAGENDR
    newData <- df %>% filter(RIAGENDR == genders)
    newData
  })
  
  output$table <- renderDataTable({
    # Get data
    Data <- getData()
    # Two-way contingency table
    contingency_table <- table(Data[, input$var1], Data[, input$var2])
    
    # Convert the contingency table to a data frame for rendering
    contingency_df <- as.data.frame.matrix(contingency_table)
    
    contingency_df
  })
  
  # Create a correlation matrix between variables
  output$correlation <- renderDataTable({
    # Create a correlation matrix between variables
    Cor_Matrix <- raw_df %>% 
      select(RIDAGEYR, BMXBMI, LBXGLU, LBXGLT, LBXIN) %>%
      cor()
    
    # Round the correlation matrix to two decimal places
    rounded_Cor_Matrix <- round(Cor_Matrix, digits = 2)
    
    # Print the rounded correlation matrix
    rounded_Cor_Matrix
  })
  
  
  output$summary <- renderDT({
    # Dynamically select the variable based on user's input
    numeric <- setdiff(names(raw_df)[sapply(raw_df, is.numeric)], "SEQN")
    selected_stat <- input$summary
    
    # Create a data frame to store summary statistics
    summary_data <- data.frame(Variable = character(), Value = numeric(), stringsAsFactors = FALSE)
    
    # Iterate through numeric columns and calculate summary statistics
    for (var in numeric) {
      value <- switch(selected_stat,
                      Q1 = round(quantile(raw_df[[var]], 0.25, na.rm = TRUE), 2),
                      Q3 = round(quantile(raw_df[[var]], 0.75, na.rm = TRUE), 2),
                      Mean = round(mean(raw_df[[var]], na.rm = TRUE), 2),
                      Median = round(median(raw_df[[var]], na.rm = TRUE), 2)
      )
      
      # Add the variable and its corresponding rounded summary statistic value to the data frame
      summary_data <- rbind(summary_data, c(var, value))
    }
    
    # Set column names
    colnames(summary_data) <- c("Variable", selected_stat)
    
    # Return the summary data frame
    summary_data
  })
  
  
  # Output plot based on user input
  output$plotType <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "Scatter Plot") {
      # both numeric variables: scatterplot
      p <- ggplot(Data, aes_string(x = input$scatx, y = input$scaty)) +
        geom_point(alpha = 0.5) +
        labs(title = paste(input$scaty, "vs.", input$scatx))
      
      
      if (input$smooth) {
        p <- p + geom_smooth(method = "lm", se = TRUE)
      }
      
      # color change
      if (input$color != "None") {
        p <- p + aes_string(color = input$color)
      }
      p
    }
  })
  
  output$plot2 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "Boxplot") {
      # one numeric var, one character var: boxplot
      p <- ggplot(Data, aes_string(x = input$boxx, y = input$boxy)) +
        geom_boxplot() +
        labs(title = paste(input$boxy, "vs.", input$boxx))
      
      # fill change
      if (input$color != "None") {
        p <- p + aes_string(fill = input$color)
      }
      
      print(p)
    }
  })
  
  output$plot3 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "BarPlot") {
      # two character variables: barplot
      p <- ggplot(Data, aes_string(x = input$barx)) +
        geom_bar(position = "dodge") +
        labs(title = paste(input$bary, "vs.", input$barx))
      # fill change
      if (input$color != "None") {
        p <- p + aes_string(fill = input$color)
      }
      print(p)
      
    }
  })
  
  output$plot4 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "Histogram") {
      # only one variable: histogram for numeric variable
      p <- ggplot(Data, aes_string(x = input$histx)) +
        geom_histogram(binwidth = 5) +
        labs(title = paste("Distribution of", input$histx))
      # fill change
      if (input$color != "None") {
        p <- p + aes_string(fill = input$color)
      }
      print(p)
      
    }
  })
}


# Run the app ----
shinyApp(ui = ui, server = server)