# Required imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Print the feature names and target names
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Print the first few samples in the dataset
print("First 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: {X[i]} (Class: {y[i]}, Species: {iris.target_names[y[i]]})")

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classification models
models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    'RandomForestClassifier': RandomForestClassifier(random_state=42)
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Test models and evaluate accuracy
accuracies = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f"Accuracy of {name}: {accuracy:.4f}")

# Find the most accurate model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"The most accurate model is: {best_model_name} with accuracy of {accuracies[best_model_name]:.4f}")

# Save the most accurate model
joblib.dump(best_model, 'best_model.joblib')
print("The best model has been saved as 'best_model.joblib'.")

# Load the saved model
loaded_model = joblib.load('best_model.joblib')
print("The model has been loaded from 'best_model.joblib'.")

# Manually enter data to test the saved model
# Example: using the first sample from the dataset
sample_data = X[0].reshape(1, -1)
scaled_sample_data = scaler.transform(sample_data)
prediction = loaded_model.predict(scaled_sample_data)
predicted_class = iris.target_names[prediction[0]]

print(f"Sample data: {sample_data}")
print(f"Predicted class: {predicted_class}")
