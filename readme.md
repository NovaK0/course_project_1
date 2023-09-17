Course Project 1 for Data Mining.
<br>
Author: Rameez Raja
<br>
Id: 202311068
<br>
Team Id: T13
<br>
The dataset information is provided in the readme file of master branch.
<br>
My work:
I have worked on all the classes and have derived the following results.
<br>
Obervations:
<br>
I have taken Data_Value_Alt feature as target, because I want to predict the percentage of surety that a value will lie in the given interval.
The highly correlated features are "Question", "Low Confidence", "High Confidence" and we are comparing rest of models trained with more features with the model trained with these three features and we call it "base_model"
<br>

-----------------------------------------------------Obervation for feature Age-------------------------------------
<br>
We call it age_model.
<br>
The overall observation from this experiment is that different age groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different age groups there was very slight change in model accuracy we can say it negligible.<br>
Because value of rmse for degree 5 for basel model is 0.043752203082452465 and for age_model is 0.04329674122386086<br>
--------------------------------------------------Obervation for feature Education--------------------------------<br>
We call it ed_model<br>
The overall observation from this experiment is that different education groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different education groups there was very slight change in model accuracy we can say it negligible.<br>
Because value of rmse for degree 5 for basel model is 0.043752203082452465 and for ed_model is 0.043318928111794296<br>

-------------------------------------------------Obervation for feature Income-----------------------------------<br>
We call it in_model<br>
The overall observation from this experiment is that different income groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different income groups there was very slight change in model accuracy we can say it negligible.<br>
Because value of rmse for degree 5 for basel model is 0.043752203082452465 and for in_model is 0.043272549387882574<br>


------------------------------------------------Obervation for feature Gender-----------------------------------<br>
We call it gen_model<br>
The overall observation from this experiment is that different gender groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different gender groups there was very slight change in model accuracy we can say it negligible.<br>
Because value of rmse for degree 5 for basel model is 0.043752203082452465 and for gen_model is 0.04373193451611556<br>

----------------------------------------------Obervation for feature Race/Ethnicity--------------------------<br>
We call it race_model<br>
The overall observation from this experiment is that different race/ethnicity groups are not significantly correlated with our Target variable "Data_Value".<br>
After training the model considering different race/ethnicity groups there was very slight change in model accuracy we can say it negligible.<br>
Because value of rmse for degree 5 for basel model is 0.043752203082452465 and for race_model is 0.061245325222929275<br>

----------------------------------------------Best features and Model--------------------------------------<br>
Best feature in support of base_model is Race/Ethnicity groups with polynomial degree 5 as it has lowest rmse of 0.04068460034731251 and highest  r2 score of 0.9999827996676635<br>
But this not that significant we can just train our model on mentioned highly three correlated features also.<br>

