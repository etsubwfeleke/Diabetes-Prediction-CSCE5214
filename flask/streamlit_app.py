import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# --- PAGE SETUP ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter your health metrics below to assess risk.")

# --- LOAD MODEL & SCALER ---
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_resources():
    model = None
    scaler = None
    
    # Load Model
    if os.path.exists('model.pkl'):
        model = pickle.load(open('model.pkl', 'rb'))
    else:
        st.error("⚠️ 'model.pkl' not found. Please upload it to your GitHub repo.")
    
    # Load Scaler (Re-creating logic based on your dataset)
    # We use a dummy dataset structure to initialize the scaler correctly
    # matching your training columns: [Glucose, Insulin, BMI, Age]
    try:
        # If you can't upload the CSV, we can manually fit the scaler 
        # using the min/max values from your original dataset if known.
        # For now, we try to load the CSV if it exists.
        if os.path.exists('diabetes.csv'):
            dataset = pd.read_csv('diabetes.csv')
            X = dataset.iloc[:, [1, 4, 5, 7]].values # Glucose, Insulin, BMI, Age
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X)
        else:
            st.warning("⚠️ 'diabetes.csv' not found. Predictions might be less accurate (Scaler missing).")
            # Fallback: Create a scaler with generic ranges
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit on theoretical maxes (Glucose 200, Insulin 800, BMI 60, Age 100)
            dummy_data = np.array([[0,0,0,0], [200, 800, 60, 100]])
            scaler.fit(dummy_data)
            
    except Exception as e:
        st.error(f"Error initializing scaler: {e}")
        
    return model, scaler

model, scaler = load_resources()

# --- INPUT FORM ---
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=30)
    
    with c2:
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

    submitted = st.form_submit_button("Analyze Risk")

# --- LOGIC ---
if submitted:
    if model is not None and scaler is not None:
        # Prepare data
        input_data = np.array([[glucose, insulin, bmi, age]])
        
        # Scale
        scaled_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        # Display Result
        st.divider()
        if prediction[0] == 1:
            st.error("### 🔴 High Risk: You may have Diabetes.")
            st.write("Please consult a healthcare professional.")
            
            # AI Suggestion Simulation
            with st.expander("See Diet Recommendations"):
                st.write("- Monitor carbohydrate intake.")
                st.write("- Increase fiber-rich foods.")
                st.write("- Exercise 30 mins daily.")
        else:
            st.success("### 🟢 Low Risk: You don't have Diabetes.")
            st.write("Keep maintaining a healthy lifestyle!")
            
            with st.expander("Prevention Tips"):
                st.write("- Maintain a balanced diet.")
                st.write("- Stay hydrated.")