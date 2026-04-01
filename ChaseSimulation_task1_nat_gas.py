#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
file_path='C:\\Users\\Devashree Buch\\Downloads\\Nat_Gas.csv'


# In[2]:


df=pd.read_csv(file_path)


# In[3]:


#df


# In[4]:


#df.to_csv('nat_gas_download.csv', index=False)


# In[5]:


import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[6]:


#prepare data
df['Dates']=pd.to_datetime(df['Dates'])
start_date=df['Dates'].min()
df['Dates']=(df['Dates']-start_date).dt.days
# X is the independent variable; date, y is the dependent variable;gas price
X=df[['Dates']]
y=df['Prices']


# In[7]:


#X,y


# In[8]:


# 2. Train-Test Split
# We set aside 20% of the data to "quiz" the model later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train) # This is where the "learning" happens


# In[9]:


prediction=model.predict(X)
prediction[0]


# In[12]:


# 3. Initialize and Train the Model
#converting the string date column into a float
# 2. Define the function to use the trained model

def get_price_for_date(input_date_str):
    # Convert the user's string to the same "Days" format used in training
    #original_start_date=df['Dates'].min()
    target_date=pd.to_datetime(input_date_str)
    days_since_start=(target_date-start_date).days
    
    # Predict for this specific single value
    # We use [[ ]] to keep it in the 2D format the model expects
    prediction = model.predict([[days_since_start]])
    
    return prediction[0]

print(get_price_for_date("2025-01-01"))
print(get_price_for_date("2026-3-30"))


# In[ ]:




