import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

# 1. Load cleaned data
df = pd.read_csv("outputs/cleaned_retail_data.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

# 2. Calculate monthly demand for each product
monthly = (
    df.groupby(["StockCode", "YearMonth"])["Quantity"]
    .sum()
    .reset_index()
)

# Include zero-sales months only after a product first appears
months = pd.period_range(
    monthly["YearMonth"].min(),
    monthly["YearMonth"].max(),
    freq="M"
)

first_month = monthly.groupby("StockCode")["YearMonth"].min()

products = first_month[first_month < months[-3]].index

full_index = pd.MultiIndex.from_tuples(
    [
        (product, month)
        for product in products
        for month in months
        if month >= first_month[product]
    ],
    names=["StockCode", "YearMonth"]
)
monthly = (
    monthly.set_index(["StockCode", "YearMonth"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

monthly = monthly.sort_values(["StockCode", "YearMonth"])

# Give each product a numeric identifier
monthly["ProductID"] = pd.factorize(monthly["StockCode"])[0]

# 3. Create previous-month demand features
for lag in [1, 2, 3]:
    monthly[f"Lag{lag}"] = (
        monthly.groupby("StockCode")["Quantity"].shift(lag)
    )

data = monthly.dropna().copy()

features = ["ProductID", "Lag1", "Lag2", "Lag3"]

# 4. Split by time, keeping test months later than training months
available_months = sorted(data["YearMonth"].unique())
split_index = int(len(available_months) * 0.8)

if split_index < 1 or split_index >= len(available_months):
    raise ValueError("Not enough monthly data for a train/test split.")

test_months = available_months[split_index:]
training_products = set(
    data[data["YearMonth"] < test_months[0]]["StockCode"]
)

test = data[
    (data["YearMonth"].isin(test_months)) &
    (data["StockCode"].isin(training_products))
]

train = data[data["YearMonth"] < test_months[0]]

X_train = train[features]
y_train = train["Quantity"]
X_test = test[features]
y_test = test["Quantity"]

# 5. Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions = predictions.clip(min=0)

# Previous month's demand is the baseline forecast
baseline = test["Lag1"].to_numpy()

# 6. Evaluate both forecasting methods
comparison = pd.DataFrame({
    "Model": ["Random Forest", "Naive Forecast"],
    "MAE": [
        mean_absolute_error(y_test, predictions),
        mean_absolute_error(y_test, baseline)
    ],
    "RMSE": [
        mean_squared_error(y_test, predictions) ** 0.5,
        mean_squared_error(y_test, baseline) ** 0.5
    ]
})

comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\nPRODUCT-LEVEL DEMAND PREDICTION")
print("--------------------------------")
print("Training months:", len(train["YearMonth"].unique()))
print("Testing months:", len(test_months))
print("Products evaluated:", test["StockCode"].nunique())
print("\nModel comparison:")
print(comparison.round(2).to_string(index=False))

# 7. Save product-level test predictions
results = test[["StockCode", "YearMonth", "Quantity"]].copy()
results = results.rename(columns={"Quantity": "ActualQuantity"})
results["PredictedQuantity"] = predictions
results["NaivePrediction"] = baseline
results["YearMonth"] = results["YearMonth"].astype(str)

results.to_csv(
    "outputs/demand_prediction_results.csv",
    index=False
)

# 8. Predict demand for the next month for each product
latest = monthly["YearMonth"].max()
next_month = latest + 1

recent = monthly[monthly["YearMonth"] >= latest - 2].copy()

forecast_rows = []

for product in products:
    history = recent[recent["StockCode"] == product].sort_values("YearMonth")

    if len(history) != 3:
        continue

    forecast_rows.append({
        "StockCode": product,
        "ProductID": int(history["ProductID"].iloc[-1]),
        "Lag1": history["Quantity"].iloc[-1],
        "Lag2": history["Quantity"].iloc[-2],
        "Lag3": history["Quantity"].iloc[-3]
    })

forecast_input = pd.DataFrame(forecast_rows)

if forecast_input.empty:
    raise ValueError("No products have enough history to forecast.")

forecast_values = model.predict(forecast_input[features]).clip(min=0)

forecast = pd.DataFrame({
    "StockCode": forecast_input["StockCode"],
    "PredictedQuantity": forecast_values.round(0).astype(int)
})

forecast["ForecastMonth"] = str(next_month)
forecast = forecast.sort_values("PredictedQuantity", ascending=False)
cutoff = forecast["PredictedQuantity"].quantile(0.75)

forecast["DemandCategory"] = forecast["PredictedQuantity"].apply(
    lambda quantity: "High Demand" if quantity >= cutoff else "Regular Demand"
)

product_names = df[["StockCode", "Description"]].drop_duplicates(
    subset=["StockCode"]
)

forecast = forecast.merge(product_names, on="StockCode", how="left")

forecast = forecast[
    ["StockCode", "Description", "ForecastMonth", "PredictedQuantity","DemandCategory"]
]

forecast.to_csv(
    "outputs/next_month_product_demand.csv",
    index=False
)


print(f"\nLatest recorded month: {latest}")
print(f"Forecast month: {next_month}")
print("\nTop 10 predicted products:")
print(forecast.head(10).to_string(index=False))

# 9. Plot total actual vs predicted demand across test months
monthly_results = results.groupby("YearMonth").agg(
    ActualDemand=("ActualQuantity", "sum"),
    PredictedDemand=("PredictedQuantity", "sum"),
    NaiveForecast=("NaivePrediction", "sum")
).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_results["YearMonth"],
    monthly_results["ActualDemand"],
    marker="o",
    label="Actual Demand"
)
plt.plot(
    monthly_results["YearMonth"],
    monthly_results["PredictedDemand"],
    marker="o",
    label="Random Forest"
)
plt.plot(
    monthly_results["YearMonth"],
    monthly_results["NaiveForecast"],
    marker="o",
    label="Naive Forecast"
)

plt.title("Actual vs Predicted Monthly Product Demand")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/demand_prediction.png")
plt.show()
plt.close()

print("\nAll prediction outputs saved successfully!")