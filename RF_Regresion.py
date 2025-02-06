import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error



# Load the dataset
file_path = "ENB2012_data.xlsx"
data = pd.read_excel(file_path)

# Define features and targets
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y_heating = data['Y1']
y_cooling = data['Y2']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for Heating Load prediction
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_scaled, y_heating, test_size=0.2, random_state=42)

# Split for Cooling Load prediction
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cooling, test_size=0.2, random_state=42)

# Train Random Forest for Heating Load
rf_model_h = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model_h.fit(X_train_h, y_train_h)
y_pred_rf_h = rf_model_h.predict(X_test_h)

# Train Random Forest for Cooling Load
rf_model_c = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model_c.fit(X_train_c, y_train_c)
y_pred_rf_c = rf_model_c.predict(X_test_c)

# Evaluate models
mse_rf_h = mean_squared_error(y_test_h, y_pred_rf_h)
r2_rf_h = r2_score(y_test_h, y_pred_rf_h)
mse_rf_c = mean_squared_error(y_test_c, y_pred_rf_c)
r2_rf_c = r2_score(y_test_c, y_pred_rf_c)

print(f"RF Heating - MSE: {mse_rf_h:.2f}, R2: {r2_rf_h:.2f}")
print(f"RF Cooling - MSE: {mse_rf_c:.2f}, R2: {r2_rf_c:.2f}")

# -------------------------------------SVM-----------------------------------------

# Load the dataset
file_path = "ENB2012_data.xlsx"
data = pd.read_excel(file_path)

# Define features and targets
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y_heating = data['Y1']
y_cooling = data['Y2']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for Heating Load prediction
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_scaled, y_heating, test_size=0.2, random_state=42)

# Split for Cooling Load prediction
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cooling, test_size=0.2, random_state=42)

# SVM Regression for Heating Load
svm_model_h = SVR(kernel='rbf', C=10, gamma=0.1)
svm_model_h.fit(X_train_h, y_train_h)
y_pred_svm_h = svm_model_h.predict(X_test_h)

# SVM Regression for Cooling Load
svm_model_c = SVR(kernel='rbf', C=10, gamma=0.1)
svm_model_c.fit(X_train_c, y_train_c)
y_pred_svm_c = svm_model_c.predict(X_test_c)

# Evaluate the models
mse_svm_h = mean_squared_error(y_test_h, y_pred_svm_h)
r2_svm_h = r2_score(y_test_h, y_pred_svm_h)
mse_svm_c = mean_squared_error(y_test_c, y_pred_svm_c)
r2_svm_c = r2_score(y_test_c, y_pred_svm_c)

print(f"SVM Heating - MSE: {mse_svm_h:.2f}, R2: {r2_svm_h:.2f}")
print(f"SVM Cooling - MSE: {mse_svm_c:.2f}, R2: {r2_svm_c:.2f}")

# -------------------------------------MLR-----------------------------------------

# Heating Load
mlr_model_h = LinearRegression()
mlr_model_h.fit(X_train_h, y_train_h)
y_pred_mlr_h = mlr_model_h.predict(X_test_h)

# Cooling Load
mlr_model_c = LinearRegression()
mlr_model_c.fit(X_train_c, y_train_c)
y_pred_mlr_c = mlr_model_c.predict(X_test_c)

# Evaluate
mse_mlr_h = mean_squared_error(y_test_h, y_pred_mlr_h)
r2_mlr_h = r2_score(y_test_h, y_pred_mlr_h)
mse_mlr_c = mean_squared_error(y_test_c, y_pred_mlr_c)
r2_mlr_c = r2_score(y_test_c, y_pred_mlr_c)

print(f"MLR Heating - MSE: {mse_mlr_h:.2f}, R2: {r2_mlr_h:.2f}")
print(f"MLR Cooling - MSE: {mse_mlr_c:.2f}, R2: {r2_mlr_c:.2f}")

# -------------------------- COMPARISON TABLE -------------------------------

results = {
    "Model": ["Random Forest", "Support Vector Machine", "Multiple Linear Regression"],
    "Heating MSE": [mse_rf_h, mse_svm_h, mse_mlr_h],
    "Heating R2": [r2_rf_h, r2_svm_h, r2_mlr_h],
    "Cooling MSE": [mse_rf_c, mse_svm_c, mse_mlr_c],
    "Cooling R2": [r2_rf_c, r2_svm_c, r2_mlr_c]
}

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Display the Table
print(comparison_df)

# Optionally, save to a CSV file
comparison_df.to_csv("model_comparison.csv", index=False)


# -------------------------- COMPARISON TABLE -------------------------------

results = {
    "Model": ["Random Forest", "Support Vector Machine", "Multiple Linear Regression"],
    "Heating MSE": [mse_rf_h, mse_svm_h, mse_mlr_h],
    "Heating R2": [r2_rf_h, r2_svm_h, r2_mlr_h],
    "Cooling MSE": [mse_rf_c, mse_svm_c, mse_mlr_c],
    "Cooling R2": [r2_rf_c, r2_svm_c, r2_mlr_c]
}

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Display the Table
print(comparison_df)

# Optionally, save to a CSV file
comparison_df.to_csv("model_comparison.csv", index=False)

# -------------------------- VISUALIZATION CODE -----------------------------
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
models = ["Random Forest", "Support Vector Machine", "Multiple Linear Regression"]
heating_mse = [mse_rf_h, mse_svm_h, mse_mlr_h]
cooling_mse = [mse_rf_c, mse_svm_c, mse_mlr_c]
heating_r2 = [r2_rf_h, r2_svm_h, r2_mlr_h]
cooling_r2 = [r2_rf_c, r2_svm_c, r2_mlr_c]

x = np.arange(len(models))

# Bar plot for MSE
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, heating_mse, width=0.4, label='Heating MSE', color='skyblue')
plt.bar(x + 0.2, cooling_mse, width=0.4, label='Cooling MSE', color='lightgreen')
plt.xlabel("Models")
plt.ylabel("Mean Squared Error")
plt.title("Model Comparison: MSE for Heating and Cooling Loads")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bar plot for R² Score
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, heating_r2, width=0.4, label='Heating R²', color='orange')
plt.bar(x + 0.2, cooling_r2, width=0.4, label='Cooling R²', color='purple')
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Comparison: R² Score for Heating and Cooling Loads")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

mape_rf_h = (abs((y_test_h - y_pred_rf_h) / y_test_h).mean()) * 100
print(f"RF Heating - MAPE: {mape_rf_h:.2f}%")

# Calculate MAPE for other models (SVR and MLR for Heating Load)
mape_svm_h = (abs((y_test_h - y_pred_svm_h) / y_test_h).mean()) * 100
mape_mlr_h = (abs((y_test_h - y_pred_mlr_h) / y_test_h).mean()) * 100

# Prepare data for plotting
models = ["Random Forest", "Support Vector Machine", "Multiple Linear Regression"]
mape_values = [mape_rf_h, mape_svm_h, mape_mlr_h]

# Plot MAPE values
plt.figure(figsize=(10, 6))
plt.bar(models, mape_values, color=['skyblue', 'orange', 'lightgreen'])
plt.title("Model Comparison: MAPE for Heating Load", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("MAPE (%)", fontsize=14)
plt.ylim(0, max(mape_values) + 5)  # Adjust y-axis limit for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#------------------------------------------------ Show ACCURACY score: -------------------------------------

threshold = 20

# Random Forest Accuracy
y_pred_rf_h_class = (y_pred_rf_h > threshold).astype(int)
y_test_h_class = (y_test_h > threshold).astype(int)
accuracy_rf_h = accuracy_score(y_test_h_class, y_pred_rf_h_class)

y_pred_rf_c_class = (y_pred_rf_c > threshold).astype(int)
y_test_c_class = (y_test_c > threshold).astype(int)
accuracy_rf_c = accuracy_score(y_test_c_class, y_pred_rf_c_class)

# SVM Accuracy
y_pred_svm_h_class = (y_pred_svm_h > threshold).astype(int)
accuracy_svm_h = accuracy_score(y_test_h_class, y_pred_svm_h_class)

y_pred_svm_c_class = (y_pred_svm_c > threshold).astype(int)
accuracy_svm_c = accuracy_score(y_test_c_class, y_pred_svm_c_class)

# MLR Accuracy
y_pred_mlr_h_class = (y_pred_mlr_h > threshold).astype(int)
accuracy_mlr_h = accuracy_score(y_test_h_class, y_pred_mlr_h_class)

y_pred_mlr_c_class = (y_pred_mlr_c > threshold).astype(int)
accuracy_mlr_c = accuracy_score(y_test_c_class, y_pred_mlr_c_class)

# Collect Results
models = ["Random Forest", "Support Vector Machine", "Multiple Linear Regression"]
accuracy_heating = [accuracy_rf_h, accuracy_svm_h, accuracy_mlr_h]
accuracy_cooling = [accuracy_rf_c, accuracy_svm_c, accuracy_mlr_c]

# Plot Results
x = np.arange(len(models))

plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, accuracy_heating, width=0.4, label='Heating Load', color='blue')
plt.bar(x + 0.2, accuracy_cooling, width=0.4, label='Cooling Load', color='green')
plt.ylim(0, 1.1)  # Accuracy ranges from 0 to 1
plt.xlabel("Models", fontsize=14)
plt.ylabel("Classification Accuracy", fontsize=14)
plt.title("Comparison of Classification Accuracy for Heating and Cooling Loads", fontsize=16)
plt.xticks(x, models, fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#------------------------------------------------ Show K-Fold scores: ---------------------------------------------------

# Perform K-Fold Cross-Validation for each model
rf_scores = cross_val_score(rf_model_h, X_scaled, y_heating, cv=3)
svm_scores = cross_val_score(svm_model_h, X_scaled, y_heating, cv=3)
mlr_scores = cross_val_score(mlr_model_h, X_scaled, y_heating, cv=3)

print(f"RF Cross-Validation Scores: {rf_scores}, Mean: {rf_scores.mean():.2f}")
print(f"SVM Cross-Validation Scores: {svm_scores}, Mean: {svm_scores.mean():.2f}")
print(f"MLR Cross-Validation Scores: {mlr_scores}, Mean: {mlr_scores.mean():.2f}")


# ------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Random Forest: Heating Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_h, y_pred_rf_h, color='blue', alpha=0.6, edgecolor='k')
plt.plot([y_test_h.min(), y_test_h.max()], [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Heating Load (Random Forest)", fontsize=16)
plt.xlabel("Actual Heating Load", fontsize=14)
plt.ylabel("Predicted Heating Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# Random Forest: Cooling Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_c, y_pred_rf_c, color='green', alpha=0.6, edgecolor='k')
plt.plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Cooling Load (Random Forest)", fontsize=16)
plt.xlabel("Actual Cooling Load", fontsize=14)
plt.ylabel("Predicted Cooling Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# MLR: Heating Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_h, y_pred_mlr_h, color='purple', alpha=0.6, edgecolor='k')
plt.plot([y_test_h.min(), y_test_h.max()], [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Heating Load (MLR)", fontsize=16)
plt.xlabel("Actual Heating Load", fontsize=14)
plt.ylabel("Predicted Heating Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# MLR: Cooling Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_c, y_pred_mlr_c, color='orange', alpha=0.6, edgecolor='k')
plt.plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Cooling Load (MLR)", fontsize=16)
plt.xlabel("Actual Cooling Load", fontsize=14)
plt.ylabel("Predicted Cooling Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# SVM: Heating Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_h, y_pred_svm_h, color='cyan', alpha=0.6, edgecolor='k')
plt.plot([y_test_h.min(), y_test_h.max()], [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Heating Load (SVM)", fontsize=16)
plt.xlabel("Actual Heating Load", fontsize=14)
plt.ylabel("Predicted Heating Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# SVM: Cooling Load
plt.figure(figsize=(10, 6))
plt.scatter(y_test_c, y_pred_svm_c, color='red', alpha=0.6, edgecolor='k')
plt.plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Cooling Load (SVM)", fontsize=16)
plt.xlabel("Actual Cooling Load", fontsize=14)
plt.ylabel("Predicted Cooling Load", fontsize=14)
plt.grid(alpha=0.3)
plt.show()


# Example for Heating Load predictions (Repeat for Cooling Load)
metrics_rf_h = [
    mean_squared_error(y_test_h, y_pred_rf_h),
    mean_absolute_percentage_error(y_test_h, y_pred_rf_h),
    r2_score(y_test_h, y_pred_rf_h)
]

metrics_mlr_h = [
    mean_squared_error(y_test_h, y_pred_mlr_h),
    mean_absolute_percentage_error(y_test_h, y_pred_mlr_h),
    r2_score(y_test_h, y_pred_mlr_h)
]

metrics_svm_h = [
    mean_squared_error(y_test_h, y_pred_svm_h),
    mean_absolute_percentage_error(y_test_h, y_pred_svm_h),
    r2_score(y_test_h, y_pred_svm_h)
]

# Organize metrics in a dictionary
heating_metrics = {
    "Random Forest": metrics_rf_h,
    "MLR": metrics_mlr_h,
    "SVM": metrics_svm_h
}

# Display the metrics in a readable format
for model, metrics in heating_metrics.items():
    print(f"Metrics for {model} (Heating Load):")
    print(f"  Mean Squared Error: {metrics[0]:.4f}")
    print(f"  Mean Absolute Percentage Error: {metrics[1]:.4f}")
    print(f"  R-squared: {metrics[2]:.4f}")
    print("-" * 40)

# Repeat this block for Cooling Load if needed
