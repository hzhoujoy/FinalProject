# FinalProject
# Joy ZHou

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(knitr)
library(caret)
library(randomForest)
library(DT)
# Dataset to be explored
raw_df <- read.csv("../NHANES_age_prediction.csv")

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
                #two columns for each of the two items
                column(6,
                       #Description of App
                       h1("What does this app do?"),
                       #box to contain description
                       box(background = "red", width = 10,
                           h4("This application presents the Exploratory Data Analysis (EDA) of the 2013-2014 NHANES (National Health and Nutrition Examination Survey) Age Prediction Subset. It showcases two models for predicting age."),
                           h4("The dataset comes from", a(href = "https://archive.ics.uci.edu", HTML("<span style='color: blue;'>UCI Machine Learning Repository</span>")),". It comprises 2,278 survey responses collected through interviews, physical examinations, and laboratory tests during the CDC’s NHANES in 2013-2014."),
                           h4("The", strong("About"), "tab provides an overview of the application, the", strong("Data Exploration"), "tab offers basic information about the dataset, and the", strong("Model"),"tab demonstrates the modeling approach applied to the dataset."),
                           h4("The primary objective of this example is to explore the relationship between",code("age group"), "and", code("creteria for diabetes"))
                       )
                ),
                
                column(6,
                       #How to use the app
                       h1("How to use the app?"),
                       #box to contain description
                       box(background = "red",width = 10,
                           h4("There are three main tabs in this app. The controls for the app are located to the left, while the visualizations are accessible on the right."),
                           h4("The", code("Data Exploration"), "tab consists of four panels, each accessible through a simple click. In the Plot Type panel, you can select the plot type and variables to visualize data distributions. The other panels provide information of numeric and character summaries."),
                           h4("To change the prior distribution, the hyperparameters can be set using the input boxes on the left.  The changes in this distribution can be seen on the first graph."),
                           h4("The", code("Modeling"), "tab consists of three subtabs. On the", code("Modeling Info"), "subtab, you can find information about logistic regression and random forest model methods. After fitting the model with your chosen predictors, explore the", code("Prediction"), "subtab to check the probability of being an adult based on various diabetes criteria.")
                       )
                     ),
                
                mainPanel(
                  imageOutput("logo")
                )
                
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
                        
                        mainPanel(
                          conditionalPanel(
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
           
           
           tabPanel("Cor_Matrix", 
                    h4(style = "color: blue; font-size: 20px;","Numeric summary is for the subsets."),
                    hr(),
                    dataTableOutput("correlation")),
           
           tabPanel("Summary", 
                    h4(style = "color: blue; font-size: 20px;","Numeric summary is for the subsets."),
                    hr(),
                    radioButtons("summary", "Numeric Summary", 
                                 choices = c("Q1", "Q3", "Mean", "Median"), 
                                 selected = "Mean"),
                    mainPanel(
                      DTOutput("summary"))
           ),
           
           tabPanel("Contigency Table",
                    h4(style = "color: blue; font-size: 20px;","The Contingency table is for the subsets."),
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
 
      # #Modeling tab content
      tabItem("hiddenmodel", ""),
      tabItem(tabName = 'model_info',
              h1("Modeling Info"),
              tabPanel("Modeling Info", 
                       fluidRow(
                         #add in latex functionality if needed
                         withMathJax(),
                         # two columns for each of the two items
                         column(6,
                                #description of modeling
                                h3("Generalized Linear Regression Model-Logistic Regression"),
                                #box to contain description
                                box(background = "light-blue", width = 12,
                                    h4("A logistic regression model is a statistical technique used to examine the relationship between a binary outcome variable and one or more predictor variables. Commonly applied in classification problems, where the objective is to predict one of two possible outcomes, usually represented as 0 and 1 or Yes and No."),
                                    h4("The logistic regression model assesses the relationship through the logistic function, also known as the sigmoid function. This function ensures that predicted probabilities fall within the range of 0 to 1."),
                                    h4("The model parameters are estimated using a process called maximum likelihood estimation. In logistic regression, the dependent variable is categorical. In our case, where the dependent variable is age_group—a binary variable, logistic regression is an appropriate method."),
                                    h4("Logistic regression provides easily interpretable results and can handle large datasets with efficience.However, It assumes a linear relationship between predictors and outcomes. If the relationship is not linear, it may not perform well. It may be prone to overfitting"),
                                    h4("The logistic regression formula is expressed as:"),
                                    h4("p", HTML("log<sub>odds</sub>(p) = β<sub>0</sub> + β<sub>1</sub>X<sub>1</sub> + β<sub>2</sub>X<sub>2</sub> + ... + β<sub>n</sub>X<sub>n</sub>")
                                     )
                                    )
                                ),
       
                              column(6,
                                
                                h3("Random Forest "),
                                #box to contain description
                                box(background = "light-blue", width = 12,
                                    h4("A random forest regression model is a versatile machine learning algorithm employed for both regression and classification tasks. It operates as an ensemble technique, utilizing multiple decision trees to formulate predictions."),
                                    h4("In this model, each decision tree is trained on a distinct subset of the data, and the final output is derived from the average of all individual decision tree outputs. This ensemble approach mitigates overfitting and enhances the overall accuracy of the model. Random forests demonstrate proficiency in predicting continuous values, such as stock prices, temperature, or sales figures, while also excelling in classifying variables."),
                                    h4("Drawbacks of the random forest model include potential issues with interpretability due to its complexity, resource-intensive training process, and the possibility of overfitting when dealing with noisy datasets."),
                                    h4("Formula for Random Forest Regression:"),
                                    h4("Random Forest Regression Formula", HTML("ŷ = Σ(Predictions from Individual Trees) / Number of Trees"))
                                    
                                )
                         ) 
                       )       
              )
      ),
    
      tabItem(tabName = 'model_fit',
              tabPanel("Model Fitting", 
                       fluidRow(
                         sidebarLayout(
                           sidebarPanel(
                             numericInput("split", "Test/Train Split Percentage", value = 0.7, min = 0.1, max = 0.9, step = 0.1),
                             selectInput("model_type", "Select Model", choices = c("Logistic Regression", "Random Forest")),
                             conditionalPanel(
                               condition = "input.model_type == 'Logistic Regression'",
                               selectInput("log_pred", "Predictor Variables for Logistic Regression", 
                                           choices = setdiff(colnames(raw_df), c("SEQN", "PAQ605", "RIDAGEYR")), multiple = TRUE)
                             ),
                             conditionalPanel(
                               condition = "input.model_type == 'Random Forest'",
                               selectInput("rf_pred", "Predictor Variables for Random Forest", 
                                           choices = setdiff(colnames(raw_df), c("SEQN", "PAQ605", "RIDAGEYR")), multiple = TRUE),
                               sliderInput("rf_cv", "Random Forest: CV Settings", min = 2, max = 10, value = 5)
                             ),
                             actionButton("fit_models", "Fit Models")
                           ),
                           
                           mainPanel(
                             tabsetPanel(
                               tabPanel("Logistic Regression", 
                                        verbatimTextOutput("model_summary_log"),
                                        textOutput("comparison_stats_log")),
                               tabPanel("Random Forest", 
                                        plotOutput("var_importance"),
                                        textOutput("comparison_stats_rf"),
                                        dataTableOutput("rf_fit_results")
                               )
                             )
                           )
                         )
                       )
                   )
              ),
    
    tabItem(tabName = 'pred',
            tabPanel("Age Group Prediction", 
                     fluidRow(
                       sidebarLayout(
                         sidebarPanel(
                           h3("Input Values"),
                           selectInput("model_type", "Select Model", choices = c("Logistic Regression", "Random Forest")),
                           sliderInput("BMXBMI", "Select BMXBMI:", min = 15,max = 60, value = 25),
                           sliderInput("LBXGLU", "Select LBXGLU:",min = 40, max = 250,value = 80),
                           sliderInput("LBXGLT", "Select LBXGLT:", min = 100, max = 300,value = 140),
                           sliderInput("LBXIN"," Select LBXIN:", min = 1, max = 100, value = 20),
                           selectInput("RIAGENDR","Select RIAGENDR:", choices = list("Female", "Male"), multiple = FALSE),
                           selectInput("DIQ010","Select DIQ010:", choices = list("Yes", "No", "Prediabetes"), multiple = FALSE),
                           conditionalPanel(
                             condition = "input.model_type == 'Logistic Regression'",
                             selectInput("log_pred_pred", "Selected Predictor Variables for Logistic Regression", 
                                         choices = NULL, multiple = TRUE)),
                           conditionalPanel(
                             condition = "input.model_type == 'Random Forest'",
                             selectInput("rf_pred_pred", "Selected Predictor Variables for Random Forest", 
                                         choices = NULL, multiple = TRUE)),
                           
                           actionButton("predict_button", "Get Predictions")
                         ),
                         # Show a plot of the generated distribution
                         mainPanel(
                           tabsetPanel(
                             tabPanel("Logistic Regression",
                                      textOutput("prediction_log"),
                                      tags$head(tags$style("#prediction_log{color: blue;
                                 font-size: 20px;
                                 font-style: italic;
                                 }")
                                      )),
                             tabPanel("Random Forest",
                                      textOutput("prediction_rf"),
                                      tags$head(tags$style("#prediction_rf{color: red;
                                       font-size: 20px;
                                       font-style: italic;
                                       }")
                                      )
                                    )
                                  )
                                )
                             )
                           )
                          )
                         )
                       )
                     )
                 )


# Define server logic ----
server <- function(input, output, session) {
  output$logo <- renderImage({
    list(src = "../R.jpg", align = "left",
         contentType = "image/jpeg", width = 600, height = 300,
         alt = "Logo")
  }, deleteFile = FALSE)

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
    # Get data
    Data <- getData()
    # Create a correlation matrix between variables
    Cor_Matrix <- Data %>% 
      select(RIDAGEYR, BMXBMI, LBXGLU, LBXGLT, LBXIN) %>%
      cor()
    
    # Round the correlation matrix to two decimal places
    rounded_Cor_Matrix <- round(Cor_Matrix, digits = 2)
    
    # Print the rounded correlation matrix
    rounded_Cor_Matrix
  })
  
  #Numeric summary
  output$summary <- renderDT({
    # Get data
    Data <- getData()
    # Dynamically select the variable based on user's input
    numeric <- setdiff(names(raw_df)[sapply(raw_df, is.numeric)], "SEQN")
    selected_stat <- input$summary
    
    # Create a data frame to store summary statistics
    summary_data <- data.frame(Variable = character(), Value = numeric(), stringsAsFactors = FALSE)
    
    # Iterate through numeric columns and calculate summary statistics
    for (var in numeric) {
      value <- switch(selected_stat,
                      Q1 = round(quantile(Data[[var]], 0.25, na.rm = TRUE), 2),
                      Q3 = round(quantile(Data[[var]], 0.75, na.rm = TRUE), 2),
                      Mean = round(mean(Data[[var]], na.rm = TRUE), 2),
                      Median = round(median(Data[[var]], na.rm = TRUE), 2)
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
      p <- ggplot(Data, aes(x = !!sym(input$scatx), y = !!sym(input$scaty))) +
        geom_point(alpha = 0.5) +
        labs(title = paste(input$scaty, "vs.", input$scatx))
      
      
      if (input$smooth) {
        p <- p + geom_smooth(method = "lm", se = TRUE)
      }
      
      # color change
      if (input$color != "None") {
        p <- p + aes(color = !!sym(input$color))
      }
      p
    }
  })
  
  output$plot2 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "Boxplot") {
      # one numeric var, one character var: boxplot
      p <- ggplot(Data, aes(x = !!sym(input$boxx), y = !!sym(input$boxy))) +
        geom_boxplot() +
        labs(title = paste(input$boxy, "vs.", input$boxx))
      
      # fill change
      if (input$color != "None") {
        p <- p + aes(fill = !!sym(input$color))
      }
      
      print(p)
    }
  })
  
  output$plot3 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "BarPlot") {
      # two character variables: barplot
      p <- ggplot(Data, aes(x = !!sym(input$barx))) +
        geom_bar(position = "dodge") +
        labs(title = paste(input$bary, "vs.", input$barx))
      # fill change
      if (input$color != "None") {
        p <- p + aes(fill = !!sym(input$color))
      }
      print(p)
      
    }
  })
  
  output$plot4 <- renderPlot({
    #get data
    Data <- getData()
    
    if (input$plotType == "Histogram") {
      # only one variable: histogram for numeric variable
      p <- ggplot(Data, aes(x = !!sym(input$histx))) +
        geom_histogram(binwidth = 5) +
        labs(title = paste("Distribution of", input$histx))
      # fill change
      if (input$color != "None") {
        p <- p + aes(fill = !!sym(input$color))
      }
      print(p)
        }
  })
# Part III_modeling
  data <- reactive({
    raw_df
  })
  print(any(is.na(raw_df))) # check missing values in raw_df
  
  # Define reactiveValues to store selected predictors for prediction
  selected_log_pred <- reactiveVal(NULL)
  selected_rf_pred <- reactiveVal(NULL)
  
  observeEvent(input$fit_models, {
    # Perform test/train split
    set.seed(123)
    split_index <- createDataPartition(y = data()$age_group, p = input$split, list = FALSE)
    train_data <- data()[split_index, ]
    test_data <- data()[-split_index, ]
    print(any(is.na(train_data))) # check missing values
    print(any(is.na(test_data))) # check missing values 
    # Prepare control parameters for train()
    ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
    
    if (input$model_type == "Logistic Regression") {
      
      # Fit logistic regression model
      log_fit <- train(age_group ~ ., 
                       data = train_data[, c("age_group", input$log_pred)], 
                       method = "glm",
                       preProcess = c("center", "scale"), 
                       family = binomial(), 
                       trControl = ctrl
      )
      
      saveRDS(log_fit, "linear_model.RDS")
      
      output$model_summary_log <- renderPrint({
        # Obtain the summary
        summary_text <- capture.output(summary(log_fit))
        # Convert the model object to a character string
        model_text <- capture.output(print(log_fit))
        # Combine both results
        result <- c(summary_text, model_text)
        # Print the combined result
        cat(result, sep = "\n")
      })
      
      # Model comparison on test set
      predictions <- predict(log_fit, newdata = test_data, type = "prob")
      predictions_numeric <- as.numeric(predictions[, "Adult"])
      
      # Convert character to numeric
      test_data$age_group_numeric <- as.numeric(factor(test_data$age_group, levels = c("Adult", "Senior")))
      
      # Check for NAs or non-numeric values in predictions_numeric
      if (any(is.na(predictions_numeric)) || any(!is.finite(predictions_numeric))) {
        # Handle the case where predictions contain NAs or non-numeric values
        output$comparison_stats <- renderText({
          "Error: Predictions contain NAs or non-numeric values"
        })
      } else {
        # Calculate AUC
        auc <- pROC::roc(test_data$age_group_numeric, predictions_numeric)
        # Calculate RMSE manually
        rmse <- sqrt(mean((test_data$age_group_numeric - predictions_numeric)^2))
        
        output$comparison_stats_log <- renderText({
          paste("AUC:", round(auc$auc, 2), "\n",
                "RMSE:", round(rmse, 2))
        })
      }
      # Store selected predictors for logistic regression model
      selected_log_pred(input$log_pred)
      selected_rf_pred(NULL)  # Reset selected predictors for random forest
      
      
    } else if (input$model_type == "Random Forest") {
      # Fit random forest model
      if (!is.null(input$rf_pred) && length(input$rf_pred) > 0) {
        rf_fit <- train(
          age_group ~ ., 
          data = train_data[, c("age_group", input$rf_pred)],
          method = "rf", 
          trControl = trainControl(method = "repeatedcv", number = input$rf_cv, repeats = 3),
          preProcess = c("center", "scale"),
          metric = "Accuracy",
          tuneGrid = data.frame(mtry = 1:ncol(train_data[, input$rf_pred]))
        )
        saveRDS(rf_fit, "Random_Forest.RDS")
        
        # Extract mtry, Accuracy, and Kappa values
        rf_results <- as.data.frame(rf_fit$results[, c("mtry", "Accuracy", "Kappa")])
        
        # Display the random forest model results in a table
        output$rf_fit_results <- renderDataTable({
          round(rf_results, 2)
        })
        
        output$rf_var_importance <- renderPlot({
          varImp(rf_fit)
        })
      } else {
        # Handle the case where no predictors are selected for the random forest
        output$rf_var_importance <- renderPlot({
          ggplot() + ggtitle("No predictors selected for Random Forest")
        })
      }
      
      # Variable Importance Plot
      output$var_importance <- renderPlot({
        varImpPlot(rf_fit$finalModel)
      })
      
      # Model comparison on test set
      predictions <- predict(rf_fit, newdata = test_data)
      predictions_numeric <- as.numeric(predictions)
      
      # Check for NAs or non-numeric values in predictions_numeric
      if (any(is.na(predictions_numeric)) || any(!is.finite(predictions_numeric))) {
        # Handle the case where predictions contain NAs or non-numeric values
        output$comparison_stats <- renderText({
          "Error: Predictions contain NAs or non-numeric values"
        })
      } else {
        
        # Convert character to numeric
        test_data$age_group_numeric <- as.numeric(factor(test_data$age_group, levels = c("Adult", "Senior")))
        
        # Now perform arithmetic operations
        rmse <- sqrt(mean((test_data$age_group_numeric - predictions_numeric)^2))
        
        output$comparison_stats_rf <- renderText({
          paste("RMSE:", round(rmse, 2))
        })
      }
      # Store selected predictors for random forest model
      selected_rf_pred(input$rf_pred)
      selected_log_pred(NULL)  # Reset selected predictors for logistic regression
      
    }
    
  })
  
  #Prediction
  # UI: Update selectInput choices based on selected model type for prediction
  observe({
    if (input$model_type == "Logistic Regression") {
      updateSelectInput(
        session,
        "log_pred_pred",
        choices = setdiff(colnames(raw_df), c("SEQN", "PAQ605", "RIDAGEYR")),
        selected = selected_log_pred()
      )
    } else if (input$model_type == "Random Forest") {
      updateSelectInput(
        session,
        "rf_pred_pred",
        choices = setdiff(colnames(raw_df), c("SEQN", "PAQ605", "RIDAGEYR")),
        selected = selected_rf_pred()
      )
    }
  })
  
  # read in the input values, and store them as a dataframe. 
  input_df <- reactive({
    data.frame(BMXBMI = input$BMXBMI,
               LBXGLU = input$LBXGLU,
               LBXGLT = input$LBXGLT,
               LBXIN = input$LBXIN,
               RIAGENDR = as.factor(input$RIAGENDR),
               DIQ010 = as.factor(input$DIQ010))
  })
  
  # Add this JavaScript function to highlight selected predictors
  shinyjs::runjs(
    "shinyjs.highlightPredictors = function(selector) {
    $(selector).css({'border': '2px solid red'});
  };"
  )
  
  # Add this JavaScript function to remove highlighting
  shinyjs::runjs(
    "shinyjs.removeHighlight = function() {
    $('input, select, textarea').css({'border': ''});
  };"
  )
  
  observeEvent(input$predict_button, {
    if (input$model_type == "Logistic Regression") {
      # read in model
      model_log <- readRDS("linear_model.RDS")
      # make the predictions
      pred <- reactive({
        # Make predictions with probabilities
        predictions_prob <- as.numeric(predict(model_log, newdata = input_df(), type = "prob"))
        predictions_prob[[1]]
      })
      
      output$prediction_log <- renderText({
        paste("The predicted probability of being an Adult by logistic regression model is:",
              round(pred(), 4) )
        })                  
    }else if (input$model_type == "Random Forest") {
      # read in model
      model_rf <- readRDS("Random_Forest.RDS")
      # make the predictions
      pred <- reactive({
        predictions_prob <- as.numeric(predict(model_rf, newdata = input_df(), type = "prob"))
        predictions_prob[[1]]
      })
      # render the prediction to HTML text for display in the mainPanel of the ui, and assign it to output$pred
      output$prediction_rf <- renderText({ 
        paste("The predicted probability of being an Adult by random frost model is:",
              round(pred(), 4))})
    }
    print(input_df())
  })
  
}

# Run the app ----
shinyApp(ui = ui, server = server)
