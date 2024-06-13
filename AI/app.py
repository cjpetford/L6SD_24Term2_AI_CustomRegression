import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the saved model
loaded_model = joblib.load('best_model.joblib')

# Streamlit UI
st.title("Iris Flower Prediction")

st.write("""
This app uses a machine learning model to predict the species of an iris flower based on its sepal and petal measurements.
""")

# Input fields for new sample data
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Predict button
if st.sidebar.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = loaded_model.predict(scaled_input_data)
    predicted_class = iris.target_names[prediction[0]]
    
    # Display prediction
    st.write(f"## Predicted Iris Species: {predicted_class}")
    
    # Display input data
    st.write("### Input Data:")
    st.write(f"Sepal Length: {sepal_length} cm")
    st.write(f"Sepal Width: {sepal_width} cm")
    st.write(f"Petal Length: {petal_length} cm")
    st.write(f"Petal Width: {petal_width} cm")

# Sidebar information
st.sidebar.write("""
### Feature Information
- **Sepal Length**: Length of the sepal in cm.
- **Sepal Width**: Width of the sepal in cm.
- **Petal Length**: Length of the petal in cm.
- **Petal Width**: Width of the petal in cm.
""")
