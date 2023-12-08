# FinalProject
# Joy ZHou

library(shiny)
library(shinydashboard)

# read in dataset
# Read the header file
nhanes_data <- read.csv("C:\\PestList\\MyDocuments\\Statistic\\ST558\\ShinyApps\\FinalProject\\NHANES_age_prediction.csv")

ui <- dashboardPage(skin="red",
                    
                    #add title
                    dashboardHeader(title="National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset", titleWidth=1000),
                    
                    #define sidebar items
                    dashboardSidebar(
                      sidebarMenu(
                      menuItem("About", tabName = "about", icon = icon("archive")),
                      menuItem("Data Exploration", tabName = "EDA", icon = icon("laptop")),
                      menuItem("Modeling", tabName = "model", icon = icon("book")),
                    )),
                    
                    #define the body of the app
                    dashboardBody(
                      tabItems(
                        # First tab content
                        tabItem(tabName = "about",
                                fluidRow(
                                  #add in latex functionality if needed
                                  withMathJax(),
                                  # Logo Section
                                  box(width = 12,
                                      height = 200,
                                      solidHeader = TRUE,
                                      img(src = "VP4_935_NHANES.png", width = "100%"),
                                      style = "overflow-x: hidden;"
                                  ),
                                  
                                  
                                  #two columns for each of the two items
                                  column(6,
                                         #Description of App
                                         h1("What does this app do?"),
                                         #box to contain description
                                         box(background="red",width=12,
                                             h4("This application presents the Exploratory Data Analysis (EDA) of the 2013-2014 NHANES (National Health and Nutrition Examination Survey) Age Prediction Subset. It showcases two models for predicting age."),
                                             h4("The dataset comes from, a(href =  from the ", a(href = "https://archive.ics.uci.edu", "UCI Machine Learning Repository"), "It comprises 2,278 survey responses collected through interviews, physical examinations, and laboratory tests during the CDC’s NHANES in 2013-2014."),
                                             h4("The **About** tab provides an overview of the application, the **Data Exploration** tab offers basic information about the dataset, and the **Model** tab demonstrates the modeling approach applied to the dataset."),
                                             h4("The primary objective of this example is to explore the relationship between age and diabetes."),
                                           )
                                  ),
                                  
                                  Column(6,
                                         #How to use the app
                                         h1("How to use the app?"),
                                         #box to contain description
                                         box(background="red",width=12,
                                             h4("The controls for the app are located to the left and the visualizations are available on the right."),
                                             h4("To change the number of successes observed (for example the number of coins landing head side up), the slider on the top left can be used."),
                                             h4("To change the prior distribution, the hyperparameters can be set using the input boxes on the left.  The changes in this distribution can be seen on the first graph."),
                                             h4("The resulting changes to the posterior distribution can be seen on the second graph.")
                                         )
                                  )
                                )
                        ),
                        #actual app layout      
                        tabItem(tabName = "EDA",
                                fluidRow(
                                  withMathJax(),
                                  column(width=3,
                                         box(width=12,background="red",sliderInput("yvalue","Y=Number of Successes",min = 0,max = 30,value = 15)
                                         ),
                                         box(width=12,
                                             title="Hyperparameters of the prior distribution for \\(\\Theta\\)",
                                             background="red",
                                             solidHeader=TRUE,
                                             p("\\(\\frac{\\Gamma(\\alpha+\\beta)}{\\Gamma(\\alpha)\\Gamma(\\beta)}\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}\\)"),
                                             h5("(Set to 1 if blank.)"),
                                             numericInput("alpha",label=h5("\\(\\alpha\\) Value (> 0)"),value=1,min=0,step=0.1),
                                             numericInput("beta",label=h5("\\(\\beta\\) Value (> 0)"),value=1,min=0,step=0.1)
                                         )
                                  ),
                                  column(width=9,
                                         fluidRow(
                                           box(width=6,
                                               plotOutput("priorPlot"),
                                               br(),
                                               h4("Prior distribution for the probability of success parameter \\(\\Theta\\).")
                                           ),
                                           box(width=6,
                                               plotOutput("distPlot"),
                                               br(),
                                               h4("Posterior distribution for the probability of success \\(\\Theta\\).")
                                           )
                                         )
                                  )
                                )
                        )
                      )
                    )
)



