#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[2]:


pip install xgboost


# In[22]:


data= pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/Bank Churn Analysis/Bank Customer Churn Prediction.csv", )


# In[23]:


data.head(10)


# In[25]:


data.info()


# # Data Exploration
# 

# In[24]:


data.describe()


# In[27]:


data.isna().sum().sum()


# In[29]:


data.duplicated().sum()


# # encode

# In[33]:


data['country'].value_counts()


# In[34]:


values= {'France': 0, 'Germany': 1, 'Spain': 2}
data.replace(values, inplace= True)


# In[35]:


data['gender'].value_counts()


# In[36]:


values= {'Male': 0, 'Female': 1}
data.replace(values, inplace= True)


# In[41]:


data.drop(['customer_id'], axis=1, inplace= True)


# In[42]:


data


# In[74]:


sns.boxplot(x="churn", y="age", data=data)


# # Checking Correlation

# In[44]:


corr=data.corr()
corr['churn']


# In[52]:


sns.heatmap(data.corr(), annot=True, annot_kws={"size": 6,}, cmap='Reds', square=True)


# # Splitting of data into training and testing , Target variable=Y is the "churn" column, which says the customer has left the bank or not (This is Binary Classification)

# In[54]:


X = data.drop(["churn"], axis=1)
y = data["churn"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Creating a model, modifying parameters using an eval set. This makes sure the model doesnt overfit with early_stop

# In[55]:


#Stops after 20 rounds without improvement
model = XGBClassifier(n_estimators=500, learning_rate=0.02, early_stopping_rounds=20) 

model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)
#Verbose=False, the loss function doesnt print each iteration

print("N_estimators early stop detected: ", model.best_iteration)


# # Checking the Accuracy, finding important features of the model

# In[56]:


predictions = model.predict(x_test)

accuracy = accuracy_score(predictions, y_test)

print(f"Accuracy of the model is {round(accuracy * 100, 4)} %")


# In[57]:


importance_scores = model.feature_importances_

importance_data = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance_scores})
importance_data = importance_data.sort_values('Importance', ascending=False)

importance_data.head(15)


# # We see that the model relies on product numbers. This can also be an error in the database. Hence, removing this column and checking the churn again.

# In[58]:


x_train2 = x_train.drop(["products_number"], axis=1)
x_test2 = x_test.drop(["products_number"], axis=1)


# In[59]:


model_2 = XGBClassifier(n_estimators=221, learning_rate=0.02, early_stopping_rounds=20)

model_2.fit(x_train2, y_train, eval_set=[(x_train2, y_train), (x_test2, y_test)], verbose=False)

print("N_estimators: ", model_2.best_iteration)


# In[61]:


predictions_2 = model_2.predict(x_test2)

accuracy_2 = accuracy_score(predictions_2, y_test)

print(f"Accuracy of the model without product no-data {round(accuracy_2 * 100, 4)} %")
print(f"Previous accuracy using the product no. column was {round(accuracy * 100, 4)} %")


# # Important Features excluding Product No.

# In[62]:


importance_scores_2 = model_2.feature_importances_

importance_data_2 = pd.DataFrame({'Importance': importance_scores_2}, index=x_train2.columns)
importance_data_2 = importance_data_2.sort_values('Importance', ascending=True)

importance_data_2.plot(kind="barh", title="Feature importance")


# # Active member, age and country are other best features to predict the churn rate.
