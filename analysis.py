import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

df = pd.read_excel("Online Retail.xlsx")

print("\n========== DATASET OVERVIEW ==========")
print("Original dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df.columns = df.columns.str.strip()

df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceNo", "StockCode", "InvoiceDate",
            "Quantity", "UnitPrice", "Country"]
)

df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["Month"] = df["InvoiceDate"].dt.month
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
df["Hour"] = df["InvoiceDate"].dt.hour

df.to_csv("outputs/cleaned_retail_data.csv", index=False)

print("\n========== CLEANING RESULTS ==========")
print("Rows after cleaning:", len(df))
print("Missing values after cleaning:")
print(df.isnull().sum())
print("Total revenue:", round(df["Revenue"].sum(), 2))
print("Unique invoices:", df["InvoiceNo"].nunique())
print("Unique products:", df["StockCode"].nunique())
print("Unique customers:", df["CustomerID"].nunique())

monthly_sales = df.groupby("YearMonth")["Revenue"].sum()

plt.figure(figsize=(12, 5))
monthly_sales.plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/charts/monthly_revenue.png")
plt.show()
plt.close()

top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
top_products.plot(kind="barh")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("outputs/charts/top_products.png")
plt.show()
plt.close()

top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
top_countries.plot(kind="barh")
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("outputs/charts/top_countries.png")
plt.show()
plt.close()

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    df.groupby("DayOfWeek")["Quantity"]
    .sum()
    .reindex(weekday_order)
)

plt.figure(figsize=(10, 5))
weekday_sales.plot(kind="bar")
plt.title("Product Quantity Sold by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/charts/weekday_demand.png")
plt.show()
plt.close()

monthly_sales.to_csv("outputs/monthly_sales.csv")
top_products.to_csv("outputs/top_products.csv")
top_countries.to_csv("outputs/top_countries.csv")
weekday_sales.to_csv("outputs/weekday_demand.csv")

print("\n========== KEY FINDINGS ==========")
print("\nTop 5 products by revenue:")
print(top_products.sort_values(ascending=False).head())

print("\nTop 5 countries by revenue:")
print(top_countries.sort_values(ascending=False).head())

print("\nMonthly revenue:")
print(monthly_sales)

print("\n========== PROJECT OUTPUTS ==========")
print("Cleaned dataset: outputs/cleaned_retail_data.csv")
print("Charts saved in: outputs/charts/")
print("Summary tables saved in: outputs/")

print("\nInitial sales analysis completed successfully!")
print("\nDate range:")
print("First date:", df["InvoiceDate"].min())
print("Last date:", df["InvoiceDate"].max())