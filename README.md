# Retail Sales Analysis and Product Demand Prediction

## Project Overview

This project focuses on analyzing retail sales data and predicting future product demand using Python and Machine Learning.

The system processes historical retail transactions, identifies sales trends, analyzes product performance, and uses a Random Forest Regression model to forecast monthly demand for individual products.

The project aims to support better understanding of sales patterns and assist with data-driven inventory planning.

## Objectives

- Clean and preprocess raw retail transaction data.
- Analyze monthly revenue and product sales patterns.
- Identify top-performing products and countries.
- Examine demand patterns across different weekdays.
- Predict monthly demand for individual products.
- Evaluate prediction performance using MAE and RMSE.
- Compare Random Forest predictions with a Naive Forecast baseline.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Dataset

**Dataset:** Online Retail Dataset

The dataset contains historical retail transaction records, including:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

## Project Workflow

### 1. Data Cleaning and Preprocessing

- Handled missing values.
- Removed duplicate and invalid transactions.
- Excluded cancelled invoices.
- Removed transactions with non-positive quantities or prices.
- Converted columns into appropriate data types.
- Calculated revenue for each transaction.

### 2. Exploratory Data Analysis

The project analyzes:

- Monthly revenue trends.
- Top 10 products by revenue.
- Top 10 countries by revenue.
- Product quantity sold by weekday.

Visualizations are generated using Matplotlib and Seaborn.

### 3. Product Demand Prediction

A Random Forest Regression model is used to predict monthly product demand.

The model uses:
- Product identification.
- Previous month's demand.
- Demand from two months ago.
- Demand from three months ago.

The dataset is divided chronologically into training and testing periods.

### 4. Model Evaluation

The Random Forest model is compared against a Naive Forecast baseline.

The evaluation metrics used are:

- **Mean Absolute Error (MAE):** Measures average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.

### Evaluation Results

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 134.55 | 979.72 |
| Naive Forecast | 133.49 | 991.48 |

The Random Forest model achieved a slightly lower RMSE,
while the Naive Forecast achieved a slightly lower MAE.
The results indicate comparable performance between the
two forecasting approaches.

### 5. Demand Forecasting

The trained model generates estimated demand for individual products for the next month.

Products are also classified into High Demand and Regular Demand categories based on their predicted quantities.

## Project Outputs

The project generates:

- Cleaned retail dataset.
- Monthly revenue summary.
- Top products and countries summaries.
- Weekday demand summary.
- Product-level prediction results.
- Random Forest and Naive Forecast comparison.
- Next-month product demand forecasts.
- Data visualization charts.

## How to Run the Project

1. Install the required libraries:

   `pip install pandas numpy matplotlib seaborn scikit-learn openpyxl`

2. Place the `Online Retail.xlsx` dataset in the project folder.

3. Run the data analysis script:

   `py analysis.py`

4. Run the prediction script:

   `py prediction.py`

5. View the generated files inside the `outputs` folder.

## Limitations

- Predictions are based on historical sales patterns and may not reflect sudden changes in customer demand.
- The model uses a limited set of historical demand features.
- Prediction accuracy depends on the availability and quality of historical data.
- The forecast is an estimate and should not be treated as a guaranteed inventory requirement.
- Further validation using additional time periods would help establish the model's reliability.

## Conclusion

This project combines retail sales analysis with machine learning-based product demand prediction. It demonstrates how Python and machine learning techniques can be applied to historical retail data to identify sales patterns and estimate future product demand.
