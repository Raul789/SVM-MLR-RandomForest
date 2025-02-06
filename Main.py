import pandas as pd

# Load the dataset
file_path = "ENB2012_data.xlsx"
data = pd.read_excel(file_path)

# Preview the data
print(data.head())

print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values (example: mean imputation)
data.fillna(data.mean(), inplace=True)


# Separate features and targets
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y_heating = data['Y1']  # Heating Load
y_cooling = data['Y2']  # Cooling Load

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split for Heating Load prediction
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_scaled, y_heating, test_size=0.2, random_state=42)

# Split for Cooling Load prediction
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cooling, test_size=0.2, random_state=42)

print("Training set shape (Heating):", X_train_h.shape, y_train_h.shape)
print("Testing set shape (Heating):", X_test_h.shape, y_test_h.shape)
print("Training set shape (Cooling):", X_train_c.shape, y_train_c.shape)
print("Testing set shape (Cooling):", X_test_c.shape, y_test_c.shape)
