import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('air_quality_model.pkl')

# Streamlit app
st.title("Air Quality Prediction App")

st.sidebar.header("Input Features")
def user_input_features():
    T = st.sidebar.slider('Temperature (T)', -20.0, 50.0, 25.0)
    RH = st.sidebar.slider('Relative Humidity (RH)', 0.0, 100.0, 50.0)
    AH = st.sidebar.slider('Absolute Humidity (AH)', 0.0, 50.0, 10.0)
    input_data = {'T': T, 'RH': RH, 'AH': AH}
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Display user inputs
st.write("### Input Features:")
st.write(input_df)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"### Predicted CO (GT): {prediction[0]:.2f}")
