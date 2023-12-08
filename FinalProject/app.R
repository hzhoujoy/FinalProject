# FinalProject
# Joy ZHou

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(knitr)
library(htmltools)
library(DT)
# Dataset to be explored
raw_df <- read.csv("C:/PestList/MyDocuments/Statistic/ST558/repos/FinalProject/NHANES_age_prediction.csv")

# Pre-compute some variables to be used by app
# convert some variable to factor
raw_df$RIAGENDR <- factor(raw_df$RIAGENDR, levels = c(1, 2), labels = c("Male", "Female"))
raw_df$PAQ605 <- factor(raw_df$PAQ605, levels = c(1, 2, 7), labels = c("Yes", "No", "Refused"))
raw_df$DIQ010 <- factor(raw_df$DIQ010, levels = c(1, 2, 3), labels = c("Yes", "No", "Prediabetes"))

character <- sapply(names(raw_df), function(x) !is.numeric(raw_df[[x]]))
character_vars <- names(raw_df)[sapply(raw_df, function(x) !is.numeric(x))]
numeric <- setdiff(names(raw_df)[sapply(raw_df, is.numeric)], "SEQN")
df <- raw_df

ui <- dashboardPage(
  
  #add title
  dashboardHeader(title = "National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset", titleWidth=1000),
  
  #define sidebar items
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("archive")),
      menuItem("Data Exploration", tabName = "EDA", icon = icon("chart-line")),
      menuItem("Modeling", tabName = "model", icon = icon("laptop"),
               menuSubItem("Modeling Info", tabName = "model_info"),
               menuSubItem("Model Fitting", tabName = "model_fit"),
               menuSubItem("Prediction", tabName = "pred")
      )
    )
  ),
  
  #define the body of the app
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "about",
              fluidRow(
                mainPanel(
                  imageOutput("logo")
                ),
                #two columns for each of the two items
                column(6,
                       #Description of App
                       h1("What does this app do?"),
                       #box to contain description
                       box(background = "light-blue", width = 12,
                           h4("This application presents the Exploratory Data Analysis (EDA) of the 2013-2014 NHANES (National Health and Nutrition Examination Survey) Age Prediction Subset. It showcases two models for predicting age."),
                           h4("The dataset comes from", a(href = "https://archive.ics.uci.edu", HTML("<span style='color: blue;'>UCI Machine Learning Repository</span>")),". It comprises 2,278 survey responses collected through interviews, physical examinations, and laboratory tests during the CDC’s NHANES in 2013-2014."),
                           h4("The **About** tab provides an overview of the application, the **Data Exploration** tab offers basic information about the dataset, and the **Model** tab demonstrates the modeling approach applied to the dataset."),
                           h4("The primary objective of this example is to explore the relationship between age and diabetes."),
                       )
                ),
                
                column(6,
                       #How to use the app
                       h1("How to use the app?"),
                       #box to contain description
                       box(background="light-blue",width=12,
                           h4("The controls for the app are located to the left and the visualizations are available on the right."),
                           h4("To change the number of successes observed (for example the number of coins landing head side up), the slider on the top left can be used."),
                           h4("To change the prior distribution, the hyperparameters can be set using the input boxes on the left.  The changes in this distribution can be seen on the first graph."),
                           h4("The resulting changes to the posterior distribution can be seen on the second graph.")
                       )
                ),
                
                
              )
      ),
      
      #actual app layout (second tab)     
      tabItem(tabName = "EDA",
              navbarPage("Dataset Explorer Analysis",
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
                                  h5(style = "color: purple; font-size: 20px;","Numeric summary is for the full data set."),
                                  hr(),
                                  radioButtons("summary", "Numeric Summary", 
                                               choices = c("Q1", "Q3", "Mean", "Median"), 
                                               selected = "Mean"),
                                  mainPanel(
                                    DTOutput("summary"))
                         ),
                         
                         tabPanel("Contigency Table",
                                  h5("Contingency table is for the full data set."),
                                  hr(),
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
      ),
      
      #Modeling tab content
      tabItem(tabName = "modeling",
              tabsetPanel(
                tabPanel("Modeling Info", textOutput("model_info")),
                tabPanel("Model Fitting", textOutput("model_fit")),
                tabPanel("Prediction", textOutput("pred"))
              )
      )
    )
  )
)




# Define server logic ----
server <- function(input, output, session) {
  
  output$logo <- renderImage({
    list(src = "C:/PestList/MyDocuments/Statistic/ST558/repos/FinalProject/R.jpg", align = "right",
         contentType = "image/jpeg", width = 600, height = 300,
         alt = "Logo")
  }, deleteFile = FALSE)
  
  
  #get data for only gender group specified
  getData <- reactive({
    genders <- input$RIAGENDR
    newData <- df %>% filter(RIAGENDR == genders)
    newData
  })
  
  # Generate data summaries
  output$summary <- renderDataTable({
    #get data
    Data <- getData()
    
    var <- input$var
    round <- input$round
    tab <- Data %>% 
      # select("Class", "InstallmentRatePercentage", var) %>%
      group_by(RIAGENDR, PAQ605, age_group) %>%
      summarize(Mean = round(mean(get(var)),round), 
                Q1 = round(quantile(get(var), 0.25), round),
                Q3 = round(quantile(get(var), 0.75), round))
    tab
  })
  
  
  output$table <- renderTable({    
    #get data
    Data <- getData()
    Data
  })
  
  # Get correlation of numeric variables
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
