# Data Mining Project Readme

## Overview
This project involves data mining and analysis using Python with various libraries such as Pandas, Seaborn, Matplotlib, NumPy, and scikit-learn. The primary goal of this project is to read, clean, and analyze a dataset, perform regression analysis, and generate visualizations to gain insights from the data.

## Database Information
- **Database Name**: Nutrition, Physical Activity, and Obesity - Behavioral Risk Factor Surveillance System (Nutrition_Physical_Activity_and_Obesity_Behavioral_Risk_Factor_Surveillance_System)
- **Source**: [CDC](https://www.cdc.gov/brfss/annual_data/annual_2019.html)
- **Description**: The dataset used in this project is sourced from the Behavioral Risk Factor Surveillance System (BRFSS) conducted by the Centers for Disease Control and Prevention (CDC). The BRFSS is a comprehensive survey that collects data related to health behaviors, risk factors, and chronic conditions among the U.S. population. It provides valuable insights into various aspects of public health, including nutrition, physical activity, obesity, and other behavioral risk factors.

## Step 1: Data Retrieval
The initial step involves retrieving the dataset from a Google Drive link. This is done using the pandas library to read the CSV file. The dataset is stored in a DataFrame, and the first few rows are displayed to provide an initial overview.

## Step 2: Data Cleaning
### Substep 2.1: Dropping Unnecessary Columns
Several columns in the dataset are dropped using the drop method. The columns to drop are specified in the columns_to_drop list. This step reduces the dimensionality of the data and removes unnecessary information.

### Substep 2.2: Handling Missing Categorical Data
A function called replace_missing_categorical is used to replace missing values in categorical columns with either a specified value or the most frequent category in that column. In this case, the column 'Income' is processed to replace missing values.

### Substep 2.3: Summary Statistics
Summary statistics are generated using the describe method to provide statistical insights into the numeric columns of the dataset.

## Step 3: Data Analysis and Visualization
### Substep 3.1: One-Hot Encoding
Categorical columns are one-hot encoded using the pd.get_dummies function, creating binary columns for each category. This is done for 'Class', 'Age(years)', 'Education', 'Gender', 'Income', and 'Race/Ethnicity' columns.

### Substep 3.2: Handling Missing Numeric Data
Missing values in the numeric columns ('Data_Value', 'Low_Confidence_Limit', and 'High_Confidence_Limit') are filled with the mean values of their respective columns.

### Substep 3.3: Outlier Removal
Outliers in the 'Data_Value' column are identified and removed based on the interquartile range (IQR). Data points beyond 1.5 times the IQR are considered outliers.

### Substep 3.4: Relationship Visualization
A function called plot_relationships_with_physical_activity is defined to visualize the relationships between 'Class_Physical Activity' and other columns. Different types of plots are generated based on the data types of the columns, including stacked bar charts and box plots.

### Substep 3.5: Correlation Heatmaps
Individual heatmaps are created to visualize the correlation between 'Class_Physical Activity' and associated columns. The correlation between these variables is shown for each associated column.

## Step 4: Regression Analysis
Regression analysis is performed to predict the 'Data_Value' column. Two models are used: linear regression and polynomial regression. The dataset is split into training and testing sets. Model performance metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2), are calculated for both regression models.

## Step 5: Results
The results of the regression analysis are displayed in a table that includes the metrics for both linear and polynomial regression models. These metrics provide insights into the accuracy and performance of the models in predicting the 'Data_Value' column.

This readme file provides an overview of the project's steps, detailed information about the database, and helps users understand the purpose and outcomes of each step in the code.