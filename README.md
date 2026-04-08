# JPMC_Forage
**JP Morgan's Quantitative Researcher Simulation**
This Forage virtual internship by J.P.Morgan Chase (https://www.theforage.com/simulations/jpmorgan/quantitative-research-11oc) consists of four main tasks-
Task 1. Extrapolating Natural Gas prices one year into the future by using appropriate forecasting methods.
Task 2. Creating a function to generate a pricing model for gas contracts by leveraging the AI model created earlier based on a set of input parameters.
Task 3. Generate a predictive model that takes loan data as an input and outputs the probability of default for a customer.
Task 4. Bucketing and rating customers based on their FICO scores in order to narrow on the probability of deafult.

Let's deep dive into these tasks!

Task 1- 
**Data**: '**Nat_Gas.csv**' is the data file for this task; consisting of two columns- Dates & Prices. 
**Analysis**: '**Tableau_data_analysis.png**' is the sample analysis image of the data. The sinusoidal wave pattern is observed in the analysis, which shows that gas utilization increases in winter (increased heat usage) and decreases during summer months thus impacting prices.
**Code**: '**ChaseSimulation_task1_nat_gas.py**' is the python notebook (IDE-JupyterNB) for this task using LinearRegression to train the model on data; see comments for explanation! (cell #11 is the code for task 2)
Note: Added **Code**: '**Chase_task1_actualVsForecast**' for Seasonal Optimization (Holt-Winters / Exponential Smoothing): Implemented a triple-exponential smoothing model via statsmodels which was overlooked by the linear regression model (used initially). This allowed for the decomposition of data into:

Level: The baseline price.
Trend: The long-term upward or downward movement.
Seasonality: The repeatable annual cycles (e.g., winter heating demand spikes). 

Included graphs to compare forecasted and actual data curve and calculated RMSE & MAE to check for errors.
Task 3- <img width="1553" height="497" alt="Screenshot 2026-04-08 142204" src="https://github.com/user-attachments/assets/c0559672-3397-410d-8701-b7ec94f7c23d" />

Task 2-In '**Chase_Simulation_task2.py**', we define price_storage_contract() function that takes in 6 parameters as input. It calculates the net value of a gas storage contract based on the parameters. For example, Profit/Loss made on buying and/or selling gas on specific dates/days/months.

**Data**: '**Task 3 and 4_Loan_Data.csv**' is the data file with following columns- customer_id, credit_lines_outstanding,loan_amt_outstanding,total_debt_outstanding,income,years_employed,fico_score,default.
**Analysis**: I performed some analysis on excel and observed that customers who have higher total_debt_outstanding than loan_amt_outstanding and credit_line>=4 have higher probability of deafulting on the loan. Lower FICO scores also play a key part.
**Code**: '**Chase_simulation_task3.py**' is the python notebook (IDE-JupyterNB) for this task. The code uses RandomForestClassifier model for classification to classify customers as default customer or not. The function predict_rf_expected_loss() defined in the code calculates propbability % and expected loss in case of default. 

Task 4-'**Chase_simulation_task4.py**' uses MSE(Mean Squared Error) [func:generate_mse_buckets() & map_rating()] to segment customers into 10 rankings based on their FICO scores. (lower the score, higher the probability of default) 

P.S. I have also uploaded example answers provided by JPMC at the end of submission.
