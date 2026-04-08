#!/usr/bin/env python
# coding: utf-8

# In[6]:


#using statsmodels instead of linear regression for predicting accurate results with minimum error rate
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[7]:


file_path=r'C:\Users\Devashree Buch\Downloads\Forage_Chase\Nat_Gas.csv' 
file_path


# In[9]:


df=pd.read_csv(file_path, parse_dates=['Dates'], index_col='Dates')
#df


# In[4]:


# 2. Prepare Data for Prophet
# Prophet REQUIRES specific column names: 'ds' for dates and 'y' for values
df_prophet = df.rename(columns={'Dates': 'ds', 'Prices': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])


# In[10]:


# Ensure data is sorted and has a monthly frequency
df = df.sort_index().asfreq('M')


# In[11]:


# 2. Fit the Model (Holt-Winters Seasonal Method)
# 'add' means the seasonality is consistent year-over-year
model = ExponentialSmoothing(df['Prices'], 
                             seasonal_periods=12, 
                             trend='add', 
                             seasonal='add').fit()


# In[14]:


# 3. Get "Fitted Values" (What the model thinks happened in the past)
# This allows you to see how accurate the model was on your existing data
df['Fitted_Values'] = model.fittedvalues


# In[15]:


# 4. Forecast for the next 12 months
forecast = model.forecast(12)


# In[16]:


# 5. Plot the Comparison
plt.figure(figsize=(12, 6))

# Plot Actual History
plt.plot(df.index, df['Prices'], label='Actual History', color='blue', marker='o', markersize=4)

# Plot Fitted Values (How the model tracked the past)
plt.plot(df.index, df['Fitted_Values'], label='Model Fit (Backtest)', color='green', linestyle='--')

# Plot Forecast (The future)
plt.plot(forecast.index, forecast, label='Future Forecast', color='red', linewidth=2)

plt.title('Natural Gas Prices: Actual vs. Model Comparison', fontsize=14)
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[18]:


#calculating RMSE & MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(df['Prices'], df['Fitted_Values'])
rmse = np.sqrt(mean_squared_error(df['Prices'], df['Fitted_Values']))

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")


# In[ ]:




