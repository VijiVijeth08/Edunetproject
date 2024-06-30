import pandas as pd

# Load the sales data
file_path = 'Smart Retail Assistant\Auto Sales data.csv'
sales_data = pd.read_csv(file_path)

# Display the first few rows
print(sales_data.head())

# Check for missing values
print(sales_data.isnull().sum())

# Fill or drop missing values as appropriate
sales_data.fillna(method='ffill', inplace=True)

# Convert date column to datetime if necessary
sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'])

# Set date as index
sales_data.set_index('ORDERDATE', inplace=True)

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fit ARIMA model
model = ARIMA(sales_data['SALES'], order=(5,1,0))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot actual sales and forecast
plt.plot(sales_data['SALES'], label='Actual Sales')
plt.plot(pd.date_range(start=sales_data.index[-1], periods=30, freq='D'), forecast, label='Forecasted Sales')
plt.legend()
plt.title('Sales Forecast')
plt.show()

# Example: Inventory management based on forecasted demand
forecasted_demand = forecast.sum()
current_inventory = 500  # Example current inventory

reorder_level = forecasted_demand - current_inventory
if reorder_level > 0:
    print(f"Reorder {reorder_level} units.")
else:
    print("Inventory level is sufficient.")

from sklearn.neighbors import NearestNeighbors
import numpy as np

# Example user-item interaction matrix (you need to prepare this from your data)
# Assume user_item_matrix is already prepared
user_item_matrix = np.array([[5, 3, 0, 1],
                             [4, 0, 0, 1],
                             [1, 1, 0, 5],
                             [1, 0, 0, 4],
                             [0, 1, 5, 4]])

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix)

user_index = 0  # Example user index
distances, indices = model_knn.kneighbors(user_item_matrix[user_index].reshape(1, -1), n_neighbors=3)

print(f"Recommended products for user {user_index}: {indices.flatten()}")

import matplotlib.pyplot as plt

# Example data visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sales_data['SALES'], label='Sales')
plt.title('Sales Trend')
plt.legend()

inventory_data = [500, 400, 300, 200, 100]  # Replace with actual inventory data
plt.subplot(1, 2, 2)
plt.bar(range(len(inventory_data)), inventory_data, label='Inventory')
plt.title('Inventory Levels')
plt.legend()

plt.show()

