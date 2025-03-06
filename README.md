### 🏥 Life Expectancy Prediction

## 📌 Project Overview

The goal of this project is, applying various regression techniques to predict life expectancy based on multiple health and socio-economic factors and 
compare different regression models and evaluate their performance using various metrics.

## 🎯 Objectives

- ✅ Predict life expectancy applying regression models
- ✅ Compare Linear, Ridge, and Lasso Regression
- ✅ Tune hyperparameters apply Grid Search and Random Search
- ✅ Visualize model performance with Prediction Error Plots
- ✅ Evaluate models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²)
- ✅ Identify underfitting or overfitting using Residuals vs. Actuals Plots

## 📂 Dataset

- **Source**: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
- **Format**: CSV
- **Key Features**:
- Life Expectancy (Target Variable)
- Country
- Year
- Status
- Percentage Expenditure
- Hepatitis B
- Population
- HIV/AIDS
- Diptheria
- Thinnes rate 1-5 years
- Thinnes rate 1-19 years
- GDP
- Adult Mortality
- Infant Deaths
- Alcohol
- Measles
- BMI
- Polio
- Total Expenditure
- Income Coposition Resources
- Schooling
- Region

## 🔧 Technologies Used

- **Python** (Jupyter Notebook)

- **Libraries**:
- 📊 Data Manipulation: pandas, numpy
- 📈 Data Visualization: matplotlib.pyplot, seaborn
- 📉 Statistical Analysis: scipy.stats🤖 Machine Learning: sklearn.linear_model (Linear, Ridge, Lasso Regression)
- 🔍 Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV
- 📏 Model Evaluation: mean_absolute_error, mean_squared_error, r2_score
- 📉 Cross Validation: K-Fold Cross Validation

## 📊 Analysis & Methodology

🔹 **Data Preprocessing & Handling**
✔️ Handled missing, duplicated values and inconsistencies
✔️ Scaled numerical features for better model performance
✔️ Encode Categorical Variables
✔️ Split the Data

🔹 **Exploratory Data Analysis (EDA)**

📌 Visualized feature distributions and correlations
📌 Analyzed trends in life expectancy across different variables
📌 Identified potential outliers and anomalies

🔹 **Model Training & Evaluation**

📍 Applied Linear, Ridge, and Lasso Regression
📍 Compared performance using MAE, RMSE, and R²
📍 Used Grid Search and Random Search for hyperparameter tuning
📍 Visualized Prediction Error Plots for model comparison
📍 Evaluated model overfitting using Residuals vs. Actuals Plots
📍 Applied K-Fold Cross Validation to improve generalization

## 📈 Results & Insights
According to the hyperparameter tuning and cross-validation outcomes, Ridge Regression was chosen as the best-performance model with moderate Alpha value of 0.3728 comparing to small 
regularisation Alpha value (0.001) of Lasso model. All of three models which are Linear, Ridge and Lasso Regressions performed almost similar metrics in terms of MSE (11.76), RMSE (3.43)
and R squared (0.86) and indicates all models are well fit for data. 
However, Ridge Regression is recommended because of its capability of dealing multicollinearity which is avoids extreme coefficient values while keeping all features. The cross-validation
outcomes also further validate the choice with an average R squared value of 0.86 through multiple folds and standard deviation value of 0.01 which is showing stable and reliable model 
evaluation. Despite Linear and Lasso models also generated same R squared value with 0.86, Ridge Regression performed better generalisation by decrease the impact of less important or 
directly related features without filtering out them, which means all independent variables contribute to the model and it leads to more stable and reliable prediction. 
Additionally, Ridge Regression performed lower variance and better trend identification for unseen data and decrease the risk of overfitting. Regarding from nature of the data, where
keeping all relevant columns is important for accuracy and prediction level.
