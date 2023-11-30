# ST558_FinalProject

This is a repo containing work from final project for ST558. 

This project uses a [National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset) from [UCI Machine Learning Repository](https://archive.ics.uci.edu) to create a Shiny App for exploratory data analysis (EDA) and predictive modeling of age-group outcomes. The primary target variable, age_group, including adult (under 65) and senior (65 years old and older). Key predictors include . EDA involves basic statistics, visualizations, correlations, and contingency tables. After EDA, the data set is split into 70% training and 30% test subsets. Two models are trained and evaluated to identify the most effective model for predicting age_group outcomes.

The following R packages were used for this project:  
+ [**`shiny`**](https://cran.r-project.org/web/packages/shiny/index.html/) for creation of apps and dashboards
+ [**`shinydashboard`**](https://cran.r-project.org/web/packages/shinydashboard/index.html) for creation of dashboards

• A line of code that would install all the packages used (so we can easily grab that and run it prior to running your app).
• The shiny::runGitHub() code that we can copy and paste into RStudio to run your app.
