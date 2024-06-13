import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge)
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt

# Define a custom regression model designed to be more accurate than linear regression
class CustomModel:
    def __init__(self):
        self.pipeline = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

# Path to Excel file
excel_file_path = r"C:\Users\icego\Desktop\Techtorium\Second Year\Projects\Term 2 24'\AI\Car_Purchasing_Data.xlsx"

# Read Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Drop irrelevant features from the original dataset
input_features = df.drop(columns=['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'])
output_feature = df['Car Purchase Amount']

# Transform input and output datasets into percentage-based weights between 0 and 1
input_scaler = MinMaxScaler()
input_scaled = input_scaler.fit_transform(input_features)

output_scaler = MinMaxScaler()
output_scaled = output_scaler.fit_transform(output_feature.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_scaled, output_scaled, test_size=0.2, random_state=42)

# Initialize and train the models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Polynomial": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Elastic Net": ElasticNet(),
    "Stepwise": RFE(estimator=LinearRegression(), n_features_to_select=5),
    "Multivariate": LinearRegression(),  # Same as Linear Regression
    "Quantile": QuantileRegressor(),
    "Bayesian": BayesianRidge(),
    "Custom Model": CustomModel()
}

for name, model in models.items():
    model.fit(X_train, y_train.ravel())
    print(f"Trained {name} model")

# Prediction on test data
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)

# Evaluate model performance
evaluation_results = {}
for name, preds in predictions.items():
    rmse = mean_squared_error(y_test, preds, squared=False)
    evaluation_results[name] = rmse
    print(f"{name} RMSE: {rmse}")

# Bar chart to visualize results
model_names = list(evaluation_results.keys())
rmse_values = list(evaluation_results.values())

plt.figure(figsize=(12, 6))
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFB3E6', '#FF6666', '#FF3388', '#66FF99', '#FF9966', '#66CC66']

bars = plt.bar(model_names, rmse_values, color=colors)
plt.xlabel('Model Used')
plt.ylabel('RMSE')
plt.title('Comparison of Regression Models')

# Add RMSE values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 4), ha='center', va='bottom')

plt.show()

# Select best model
best_model_name = min(evaluation_results, key=evaluation_results.get)
best_model = models[best_model_name]
print(f"Best model: {best_model_name} with RMSE: {evaluation_results[best_model_name]}")

# Gather user inputs
user_inputs = np.array([
    [1, 35, 75000, 10000, 500000]
])

# Transform user inputs using the same scaler
user_inputs_scaled = input_scaler.transform(user_inputs)

# Use model to make predictions based on user input
user_predictions_scaled = best_model.predict(user_inputs_scaled)
user_predictions = output_scaler.inverse_transform(user_predictions_scaled.reshape(-1, 1))
print("\nPredicted CPA for user inputs:", user_predictions)

# Rounded prediction for easy reading 
rounded_prediction = np.round(user_predictions, -3)
print(f"\n\nThe Predicted Car Purchase Amount is around: ${rounded_prediction[0, 0]}\n")

# Calculate and print accuracy difference
linear_rmse = evaluation_results["Linear"]
custom_rmse = evaluation_results["Custom Model"]
accuracy_difference = linear_rmse - custom_rmse
print(f"Accuracy Difference: {accuracy_difference}")
