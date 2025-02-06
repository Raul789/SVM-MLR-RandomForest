from sklearn.linear_model import LinearRegression

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
