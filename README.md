we have got dataset of Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System of size 53392 instances and 33 columns,<br>
which have different classes like,<br>
1.Obesity / Weight Status<br>
2.Fruits and Vegetables<br>
3.Physical Activity.<br>

I am working on the specific class which is "Obesity / Weight Status".<br>
I have taken Data_Value feature as target, because I want to predict the percentage of surety that a value will lie in the given interval for the particular class "Obesity / Weight Status"<br>
I have perform data cleaning, dropping duplicates column, dropping columns which had more than 70 percentage missing value.<br>

I have performed EDA, plotting graph between different features of class "Obesity / Weight Status" like pie chart, histogram, box plot etc...<br>
then i have show the heatmap for getting correlation between those features.<br>

And performing the linear regression and polynomial regression with degree 5 of base model.<br>
Performing the linear regression and polynomial regression with degree 5 of base model between base model and age model.<br>
Performing the linear regression and polynomial regression with degree 5 of base model between base model and education model.<br>
Performing the linear regression and polynomial regression with degree 5 of base model between base model and gender model.<br>
Performing the linear regression and polynomial regression with degree 5 of base model between base model and income model.<br>
Performing the linear regression and polynomial regression with degree 5 of base model between base model and Race/Ethnicity model.<br>


Observations And Conclusion:<br>
I have taken Data_Value feature as target, because I want to predict the percentage of surety that a value will lie in the given interval for the particular class "Obesity / Weight Status".<br>
The highly correlated features are "Question", "Low Confidence", "High Confidence".<br>
With the model trained with these three features and we call it "base_model"<br>

Observation for feature Age -<br>
We call it age_model.<br>
From the observation we can depict that different age groups are not much correlated with target variable data_value.<br>
After training the model considering different age groups there was a very slight change in model accuracy we can say it's negligible.<br>
Because the value of RMSE for degree 5 for base model is 0.04523233522774564 and for age_model is 0.04574244247072131.<br>

Observation for feature Education -<br>
We call it ed_model.<br>
The overall observation from this experiment is that different education groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different education groups there was a very slight change in model accuracy we can say it's negligible.<br>
Because the value of RMSE for degree 5 for base model is 0.04523233522774564 and for ed_model is 0.04499433244988705.<br>

Observation for feature Gender -<br>
We call it gen_model.<br>
The overall observation from this experiment is that different gender groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different gender groups there was a very slight change in model accuracy we can say it's negligible.<br>
Because the value of RMSE for degree 5 for base model is 0.04523233522774564 and for gen_model is 0.04893917721678798.<br>

Observation for feature Income -<br>
We call it in_model.<br>
The overall observation from this experiment is that different income groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different income groups there was a very slight change in model accuracy we can say it's negligible.<br>
Because the value of RMSE for degree 5 for base model is 0.04523233522774564 and for in_model is 0.044509896120144374.<br>

Observation for feature Race/Ethnicity -<br>
We call it race_model.<br>
The overall observation from this experiment is that different race/ethnicity groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different race/ethnicity groups there was a very slight change in model accuracy we can say it's negligible.<br>
Because the value of RMSE for degree 5 for base model is 0.04523233522774564 and for race_model is 0.03895023435254968.<br>
