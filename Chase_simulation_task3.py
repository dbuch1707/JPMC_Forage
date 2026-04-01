#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


path='C:\\Users\\Devashree Buch\\Downloads\\Forage_Chase\\Task 3 and 4_Loan_Data.csv'


# In[21]:


df=pd.read_csv(path)


# In[4]:


#df.head()


# In[5]:


#df.columns


# In[6]:


#Defining feature & vector columns
X=df[['credit_lines_outstanding', 'loan_amt_outstanding',
       'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y=df['default']


# In[7]:


#X,y


# In[8]:


#Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


#Feature Scaling (for classification performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


#Initialize and Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# In[11]:


#Make Predictions
predictions = model.predict(X_test)


# In[18]:


#Evaluate the Results
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[22]:


#testing on sample data
test_case = pd.DataFrame([[5, 5000, 8000, 75000, 5, 800]], columns=['credit_lines_outstanding', 'loan_amt_outstanding',
       'total_debt_outstanding', 'income', 'years_employed', 'fico_score'])


# In[23]:


#scaling the dataframe
#scaled_test_input = scaler.transform(test_case)

#Call the prediction function
final_prediction = model.predict(test_case)

#Output the result
print(f"The model classifies this data as: {final_prediction[0]}")


# In[15]:


print(df.dtypes)


# In[24]:


import pandas as pd
import numpy as np

def predict_rf_expected_loss(credit_lines, loan_amt, total_debt, income, years_emp, fico):
    """
    Uses the existing Random Forest model to calculate Expected Loss.
    """
    # 1. Format the input to match the training data columns exactly
    feature_names = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                     'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
    
    input_df = pd.DataFrame([[credit_lines, loan_amt, total_debt, income, years_emp, fico]], 
                            columns=feature_names)
    
    # 2. Preprocess
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict Probability of Default (PD)
    # predict_proba returns [prob_class_0, prob_class_1]. We want index 1 (Default).
    pd_probabilities = model.predict_proba(input_scaled)
    pd_value = pd_probabilities[0][1] 
    
    # 4. Financial Constants
    recovery_rate = 0.10
    lgd = 1 - recovery_rate  # Loss Given Default (0.90)
    
    # 5. Calculate Expected Loss (EL = PD * EAD * LGD)
    # EAD (Exposure at Default) is the total_debt_outstanding
    expected_loss = pd_value * total_debt * lgd
    
    return expected_loss, pd_value

# --- Example Usage ---
el, pd = predict_rf_expected_loss(5, 5000, 8000, 75000, 5, 800)

print(f"Random Forest PD: {pd:.2%}")
print(f"Calculated Expected Loss: ${el:,.2f}")


# In[ ]:




