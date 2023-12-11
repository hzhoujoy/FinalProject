# ST558_FinalProject

This is a repo containing work from final project for ST558. 

This project uses a [National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset) from [UCI Machine Learning Repository](https://archive.ics.uci.edu) to create a Shiny App for exploratory data analysis (EDA) and predictive modeling of age-group outcomes. The primary target variable, age_group, including adult (under 65) and senior (65 years old and older). Key predictors include . EDA involves basic statistics, visualizations, correlations, and contingency tables. After EDA, the data set is split into training and 3test subsets. Two models are trained and evaluated to identify the most effective model for predicting age_group outcomes.

The following R packages were used for this project:  
+ [**`shiny`**](https://cran.r-project.org/web/packages/shiny/index.html/) Provides an elegant and powerful web framework for creation of apps and dashboards
+ [**`shinydashboard`**](https://cran.r-project.org/web/packages/shinydashboard/index.html) for creation of dashboards
+ [**`tidyverse`**](https://www.tidyverse.org/) An opinionated collection of R packages designed for data science.
+ [**`caret`**](https://cran.r-project.org/web/packages/caret/) A set of functions that attempt to streamline the process for creating predictive models.  
+ [**`ggplot2`**](https://ggplot2.tidyverse.org/) for creating graphics. 
+ [**`DT`**](https://rstudio.github.io/DT/) Provides an R interface to the JavaScript library Data Tables.  
+ [**`knitr`**](https://cran.r-project.org/web/packages/knitr/index.html) is a toll for dynamic report results generated with R.
+ [**`glmnet`**](https://cran.r-project.org/web/packages/glmnet/index.html) A package that fits generalized linear and similar models via penalized maximum likelihood.
+ [**`LiblineaR`**](https://cran.r-project.org/web/packages/LiblineaR/index.html)  Provides a simple library for solving large-scale regularized linear classification and regression problems.  
+ [**`pls`**](https://cran.r-project.org/web/packages/pls/index.html) Provides functions for performing Partial Least Squares analysis.


Install the required packages  
```
install.packages(c('shiny', 'shinydashboard', 'tidyverse', 'caret', 'ggplot2', 'knitr', 'DT', 'metrics', 'nnet', 'dplyr', 'lattice'))
```

The shiny::runGitHub() code that we can copy and paste into RStudio to run the app  

```
shiny::runGitHub("hzhoujoy/FinalProject")

```