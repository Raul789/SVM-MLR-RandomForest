import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

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
