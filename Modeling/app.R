# modeling
# A multiple linear regression or generalized linear regression model
# A random forest model

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(knitr)
#library(htmltools)
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
      tabItem(tabName='model_info',
              h1("Modeling Info"),
              tabPanel("Modeling Info", 
                       fluidRow(
                         #add in latex functionality if needed
                         withMathJax(),
                         # two columns for each of the two items
                         column(6,
                                #description of modeling
                                h1("Generalized Linear Regression Model-Logistic Regression"),
                                #box to contain description
                                box(background = "light-blue", width = 12,
                                    h4(""),
                                    h4(""),
                                    h4(""),
                                    h4("")
                                )
                         ),
                         
                         column(6,
                                #How to use the app
                                h1("Random Forest "),
                                #box to contain description
                                #box to contain description
                                box(background = "light-blue", width = 12,
                                    h4(""),
                                    h4(""),
                                    h4(""),
                                    h4("")
                                )
                         ) 
                       )       
              ),
      ),
      tabItem(tabName='model_fit',
              h1("Model Fitting"),
              tabPanel("Model Fitting", textOutput("model_fit")),
      ),
      tabItem(tabName='pred',
              h1("Prediction"),
              tabPanel("Prediction", textOutput("pred"))      
      )
    )
  )
)


          #

# server logic---
server <- function(input, output, session) {}



# Run the application 
shinyApp(ui = ui, server = server)
