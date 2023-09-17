## Course-Project-1 of Introduction to Data Mining

The dataset focuses on people's health status, with obesity being a risk if they don't get enough nutrients or engage in physical exercise. To analyze the data, we observe unnecessary columns and remove them, determining the number of null values. We plot a Piechart for the Class column to see the percentages of different labels, such as Fruits and Vegetables, Physical Activity, and Obesity/Weight Status.

For the Question column, we plot the number of people falling under labels like Low Fruit Consumption, Overweight, or No Physical Activity. We drop the Physical Activity and Obesity/Weight Status class labels and plot a barplot of the LocationDesc column to see from how many different cities the data is collected. We perform one-hot encoding on Age, Income, Education, and Race\Ethnicity columns, and Label Encoding on categorical data.

Heatmaps were drawn to find correlations between columns, but there was a low correlation. A boxplot for Low_Confidence_Limit and High_Confidence_Limit showed most values falling between 20 to 35 and 28 to 45, with some outliers.

We trained our model by including all features, then focusing on Age, Income, and Education features. We found that there is no significant difference in Train_RMSE, Test_RMSE, Train_MAE, and Test_MAE for different datasets, allowing us to consider any of them.
