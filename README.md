# ST558_FinalProject

## This is a repo containing work from final project for ST558. 

This project uses a [National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset) from [UCI Machine Learning Repository](https://archive.ics.uci.edu) to create a Shiny App for exploratory data analysis (EDA) and predictive modeling of age-group outcomes. The primary target variable, age_group, including adult (under 65) and senior (65 years old and older). Key predictors include . EDA involves basic statistics, visualizations, correlations, and contingency tables. After EDA, the data set is split into training and 3test subsets. Two models are trained and evaluated to identify the most effective model for predicting `age_group` outcomes. In the app, we can choose the predictors and use the values of predictors to do predictions.

## Description of variables in the data set:   
+ **SEQN:** Respondent Sequence Number (ID)  
+ **age_group:** Age Group (Senior/Adult)  
+ **RIDAGEYR:** Age   
+ **RIAGENDR:** Gender 1 = Male 2  = Female
+ **PAQ605:** takes part in weekly physical activity 1 = yes 2 = no 7 = Refused  
+ **BMXBMI:** Body Mass Index 
+ **LBXGLU:** Blood Glucose after fasting-Fasting Glucose (mg/dL)  
+ **DIQ010:** Doctor told you have diabetes 1 = yes 2 = no 3 = Borderline or Prediabetes (blood sugar is higher than normal but not high enough to be called diabetes or sugar diabetes)
+ **LBXGLT:** Oral Glucose Tolerance Test - Two Hour Glucose(OGTT)(mg/dL)   
+ **LBXIN:** Blood Insulin Levels(uU/mL)

## The following R packages were used for this project:  
+ [**`shiny`**](https://cran.r-project.org/web/packages/shiny/index.html/) Provides an elegant and powerful web framework for creation of apps and dashboards  
+ [**`shinydashboard`**](https://cran.r-project.org/web/packages/shinydashboard/index.html) for creation of dashboards.  
+ [**`tidyverse`**](https://www.tidyverse.org/) An opinionated collection of R packages designed for data science.  
+ [**`caret`**](https://cran.r-project.org/web/packages/caret/) A set of functions that attempt to streamline the process for creating predictive models.  
+ [**`ggplot2`**](https://ggplot2.tidyverse.org/) for creating graphics. 
+ [**`DT`**](https://rstudio.github.io/DT/) Provides an R interface to the JavaScript library Data Tables.  
+ [**`knitr`**](https://cran.r-project.org/web/packages/knitr/index.html) is a toll for dynamic report results generated with R.  
+ [**`dplyr`**](https://cran.r-project.org/web/packages/dplyr/index.html) is a powerful tool for working with data frame.  
+ [**`lattice`**](https://cran.r-project.org/web/packages/lattice/index.html) is a powerful graphics package for creating and visualizing various types of statistical graphics.  
+ [**`randomForest`**](https://cran.r-project.org/web/packages/randomForest/index.html) provides functions for building a Random Forest model.   
+ [**`shinyjs`**](https://cran.r-project.org/web/packages/shinyjs/index.html) performs JavaScript operations in shiny apps.    

## Install the required packages  
```
install.packages(c("shiny", "shinydashboard", "tidyverse", "caret", "ggplot2", "DT", "knitr", "dplyr", "lattice","randomForest", "shinyjs"))
```

## The shiny::runGitHub() code for runing in the Rstudio  

```
shiny::runGitHub("FinalProject", "hzhoujoy", subdir = "FinalProject")

runUrl("https://github.com/hzhoujoy/FinalProject/archive/HEAD.tar.gz",
       subdir = "FinalProject")
# Run the app ----
shinyApp(ui = ui, server = server)

```
