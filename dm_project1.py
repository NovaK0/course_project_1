#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



import math


# In[2]:


data=pd.read_csv(r"C:\Users\DHARM\Desktop\DM_Project\Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv")


# In[3]:


data


# In[4]:


missing_value = data.isnull().sum()
missing_value


# In[5]:


data.info()


# In[6]:


col = ['Unnamed: 0','LocationAbbr','Class','Data_Value_Unit','Data_Value_Type','ClassID']
data.drop(columns = col, inplace = True)


# In[7]:


data


# In[8]:


data.info()


# In[9]:


col=['Data_Value_Footnote_Symbol','Data_Value_Footnote','DataValueTypeID','GeoLocation']
data.drop(columns = col, inplace = True)


# In[10]:


data


# In[11]:


data.info()


# In[12]:


col=['QuestionID','LocationID','StratificationCategory1','Stratification1','StratificationCategoryId1','StratificationID1']
data.drop(columns=col,inplace = True)


# In[13]:


data


# In[14]:


data.info()


# In[15]:


col=['Total','TopicID']
data.drop(columns=col,inplace=True)


# In[16]:


data


# In[17]:


col=['Datasource','Data_Value_Alt']
data.drop(columns=col,inplace=True)


# In[18]:


data


# In[19]:


data.info()


# In[20]:


x= data.Question.value_counts()


# In[21]:


plt.figure(figsize=(20, 10))
plt.pie(x, labels=x.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Distribution of Categories')
plt.show()


# In[22]:



sns.boxplot(x="LocationDesc", y="Data_Value", data=data) 
plt.xlabel("LocationDesc")
plt.ylabel("Data_Value")
plt.title("Topic")
plt.xticks(rotation=90)
plt.show()


# In[23]:


sns.barplot(x='Topic', y='Data_Value', data=data)
plt.xticks(rotation=90)
plt.show()


# In[24]:


data.info()


# In[25]:


data['Topic'].value_counts()


# In[26]:


data['Topic'].value_counts().plot(kind='bar')
plt.xlabel('Topics')
plt.ylabel('Counts')
plt.title('Topic Distribution')
plt.show()


# In[27]:


data['Question'].value_counts().plot(kind='bar')
plt.xlabel('Question')
plt.ylabel('Counts')
plt.title('counts the Questions')
plt.show()


# In[28]:


Drop_val ='Physical Activity - Behavior'
filtered_data = data[data['Topic'] != Drop_val ]
unique_value = filtered_data['Topic'].nunique()
print(unique_value)


# In[29]:


Drop_val ='Fruits and Vegetables - Behavior'
filtered_data_new =filtered_data[filtered_data['Topic'] != Drop_val ]
unique_value = filtered_data_new['Topic'].nunique()
print(unique_value)


# In[30]:


filtered_data_new.columns


# In[31]:


topic = filtered_data_new['Topic'].value_counts()
topic


# In[32]:


import numpy as np
numeric_features=data.select_dtypes(include=[np.number])
numeric_features.columns


# In[33]:


categorical_features=data.select_dtypes(include=[np.object])
categorical_features.columns


# In[34]:


correlation= numeric_features.corr()
print(correlation['Low_Confidence_Limit'].sort_values(ascending=False),'\n')


# In[35]:


from sklearn.preprocessing import LabelEncoder
categorical_columns = ['LocationDesc', 'Topic', 'Question']

label_encoder = LabelEncoder()

for column in categorical_columns:
    filtered_data_new[column] = label_encoder.fit_transform(filtered_data_new[column])

filtered_data_new.info()


# In[36]:


filtered_data_new = pd.get_dummies(filtered_data_new, columns=['Age(years)', 'Education','Gender','Income','Race/Ethnicity'], prefix=['Age(years)', 'Education','Gender','Income','Race\Ethnicity'])


# In[37]:


filtered_data_new.info()


# In[38]:


filtered_data_new


# In[39]:


filtered_data_new.isnull().sum()


# In[40]:


# Calculating Mean, Mode and Median for feature Data_Value_Alt and we can see that it has almost same mean and median values also it is bimodal.
# But mean > median > mode
# This means that the distribution is bit positvely skewed.
# So we can replace the missing values with median for this feature.
print("Mean", data['Data_Value'].mean())
print("Median", data['Data_Value'].median())
print("Mode", data['Data_Value'].mode())


# In[41]:


# Calculating Mean, Mode and Median for feature Low_Confidence_Limit and we can see that it has almost same mean, mode and median values.
# But here mode > mean
# This means that the distribution is bit negatively skewed.
# So we can replace the missing values with median for this feature.
print("Mean", data['Low_Confidence_Limit'].mean())
print("Median", data['Low_Confidence_Limit'].median())
print("Mode", data['Low_Confidence_Limit'].mode())


# In[42]:



print("Mean", data['Sample_Size'].mean())
print("Median", data['Sample_Size'].median())
print("Mode", data['Sample_Size'].mode())


# In[43]:


print("Mean", data['High_Confidence_Limit '].mean())
print("Median", data['High_Confidence_Limit '].median())
print("Mode", data['High_Confidence_Limit '].mode())


# In[44]:


columns_to_fillna = ['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit ','Sample_Size']

for column in columns_to_fillna:
    if filtered_data_new[column].dtype == 'float64' or filtered_data_new[column].dtype == 'int64':
        mean_value = data[column].mean()
        filtered_data_new[column].fillna(mean_value, inplace=True)


# In[45]:


filtered_data_new.isnull().sum()


# In[46]:


correlation_columns = [
  'Question',
    'Data_Value',
    'Low_Confidence_Limit',
    'High_Confidence_Limit ',
    'Income_$15,000 - $24,999',
    'Income_$25,000 - $34,999',
    'Income_$35,000 - $49,999',
    'Income_$50,000 - $74,999',
    'Income_$75,000 or greater',
    'Income_Less than $15,000'
]
correlation_matrix1 = filtered_data_new[correlation_columns].corr()
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', linewidths=0.8)
plt.title('Correlation of Income')
plt.show


# In[47]:


correlation_columns = [
 'YearEnd', 'LocationDesc', 'Question', 'Low_Confidence_Limit', 'High_Confidence_Limit ',
            'Education_College graduate', 'Education_High school graduate','Education_Less than high school',
            'Education_Some college or technical school'
]
correlation_matrix1 = filtered_data_new[correlation_columns].corr()
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', linewidths=0.8)
plt.title('Correlation of Education')
plt.show


# In[48]:


# TRAINING THE MODEL WITH THE Income PARAMETER


# In[49]:


income_data = filtered_data_new[['YearEnd', 'LocationDesc', 'Topic', 'Question', 'Low_Confidence_Limit', 'High_Confidence_Limit ',
            'Income_$15,000 - $24,999', 'Income_$25,000 - $34,999','Income_$35,000 - $49,999',
            'Income_$50,000 - $74,999', 'Income_$75,000 or greater', 'Income_Less than $15,000']]

income_data.head()


# In[50]:


X_train, X_test, y_train, y_test =train_test_split(income_data, filtered_data_new["Data_Value"], random_state=42, test_size=0.20)


# In[51]:


X_train


# In[52]:


y_train


# In[53]:


size_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = size_scaler.transform(X_train)
X_test_scaled = size_scaler.transform(X_test)


# In[54]:


X_train_scaled


# In[55]:


X_train_scaled.shape, X_test_scaled.shape


# In[56]:


poly_regression = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

model_dict = {
    'LinearRegression': {"model": LinearRegression(), "params": {}},
    'PolynomialRegression': {"model": poly_regression, "params": {}}
}


# In[57]:


def eval_models():
    models_result = pd.DataFrame()
    models_result['Train_RMSE'] = None
    models_result['Test_RMSE'] = None
    models_result['Train_MAE'] = None
    models_result['Test_MAE'] = None
    models_result['best_params'] = None

    best_reg_model_ours = None
    best_test_score = math.inf

    for model_name, reg_model in model_dict.items():
        classifier = GridSearchCV(reg_model['model'], reg_model['params'], n_jobs=20, verbose=0)
        classifier.fit(X_train_scaled, list(y_train))
        best_model = classifier.best_estimator_

        y_train_predicted = best_model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(list(y_train), y_train_predicted))
        train_mae = mean_absolute_error(list(y_train), y_train_predicted)

        print(model_name, train_rmse, classifier.best_params_)

        y_predicted = best_model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(list(y_test), y_predicted))
        test_mae = mean_absolute_error(list(y_test), y_predicted)

        if test_rmse < best_test_score:
            best_test_score = test_rmse
            best_reg_model_ours = best_model

        models_result.loc[model_name, ['Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'best_params']] = [train_rmse, test_rmse, train_mae, test_mae, classifier.best_params_]

    print("Best model: ", best_model)
    # plot the prediction errors using the best model
    y_predicted = best_model.predict(X_test_scaled)
    plt.plot(list(y_test) - y_predicted, marker='o', linestyle='')

    return models_result


# In[58]:


models_result = eval_models()


# In[59]:


models_result


# In[60]:


#TRAINING THE MODEL WITH THE Education PARAMETER


# In[62]:


Education_data = filtered_data_new[['YearEnd', 'LocationDesc', 'Topic', 'Question', 'Low_Confidence_Limit', 'High_Confidence_Limit ',
            'Education_College graduate', 'Education_High school graduate','Education_Less than high school',
            'Education_Some college or technical school']]

Education_data.head()


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(Education_data, filtered_data_new["Data_Value"], random_state=42, test_size=0.20)


# In[64]:


X_train


# In[65]:


y_train


# In[66]:


size_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = size_scaler.transform(X_train)
X_test_scaled = size_scaler.transform(X_test)


# In[67]:


X_train_scaled


# In[68]:


X_test_scaled


# In[69]:


X_train_scaled.shape, X_test_scaled.shape


# In[70]:


poly_regression = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

model_dict = {
    'LinearRegression': {"model": LinearRegression(), "params": {}},
    'PolynomialRegression': {"model": poly_regression, "params": {}}
}


# In[71]:


def eval_models():
    models_result = pd.DataFrame()
    models_result['Train_RMSE'] = None
    models_result['Test_RMSE'] = None
    models_result['Train_MAE'] = None
    models_result['Test_MAE'] = None
    models_result['best_params'] = None

    best_reg_model_ours = None
    best_test_score = math.inf

    for model_name, reg_model in model_dict.items():
        classifier = GridSearchCV(reg_model['model'], reg_model['params'], n_jobs=20, verbose=0)
        classifier.fit(X_train_scaled, list(y_train))
        best_model = classifier.best_estimator_

        y_train_predicted = best_model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(list(y_train), y_train_predicted))
        train_mae = mean_absolute_error(list(y_train), y_train_predicted)

        print(model_name, train_rmse, classifier.best_params_)

        y_predicted = best_model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(list(y_test), y_predicted))
        test_mae = mean_absolute_error(list(y_test), y_predicted)

        if test_rmse < best_test_score:
            best_test_score = test_rmse
            best_reg_model_ours = best_model

        models_result.loc[model_name, ['Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'best_params']] = [train_rmse, test_rmse, train_mae, test_mae, classifier.best_params_]

    print("Best model: ", best_model)
    # plot the prediction errors using the best model
    y_predicted = best_model.predict(X_test_scaled)
    plt.plot(list(y_test) - y_predicted, marker='o', linestyle='')

    return models_result


# In[72]:


models_result = eval_models()


# In[73]:


models_result


# In[ ]:




