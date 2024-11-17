# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(r"C:/Users/Nikhil/Downloads/archive/House_Price_Prediction_Dataset.csv")

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())
print("\nColumns in the dataset:", data.columns)

missing_values = data[['Condition', 'Price']].isnull().sum()
print("\nMissing values in 'Condition' and 'Price':")
print(missing_values)
data = data.dropna(subset=['Condition', 'Price'])

# Plotting a box plot for Condition vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(x="Condition", y="Price", data=data)
plt.title("Box Plot of Condition vs Price")
plt.show()

# Plotting a histogram for Price
plt.figure(figsize=(10, 6))
plt.hist(data['Price'], bins=30, edgecolor='black')
plt.title("Histogram of Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Scatter plot to visualize Price vs Area
plt.figure(figsize=(10, 6))
plt.scatter(data['Area'], data['Price'], alpha=0.5)
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Calculate and display Pearson correlation for numeric columns only
numeric_data = data.select_dtypes(include=[np.number])  
correlation_matrix = numeric_data.corr()  
print("\nCorrelation Matrix:")
print(correlation_matrix['Price'].sort_values(ascending=False))

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Identify independent variables and verify presence of each
features = ['Condition', 'Area', 'Bedrooms', 'Bathrooms']
if 'Garage' in data.columns:
    features.append('Garage') 

categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
print("Columns after encoding:", data_encoded.columns)

# Define dependent and independent variables
X = data_encoded[['Area', 'Bedrooms', 'Bathrooms'] + [col for col in data_encoded.columns if 'Condition_' in col]]
if 'Garage' in data_encoded.columns:
    X['Garage'] = data_encoded['Garage']

y = data_encoded['Price']  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions on test set:", y_pred[:5])

# Calculate and display the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error on Test Set:", mse)

# Display first 5 actual vs predicted values for comparison
comparison = pd.DataFrame({'Actual': y_test[:5].values, 'Predicted': y_pred[:5]})
print("\nActual vs Predicted Prices:")
print(comparison)