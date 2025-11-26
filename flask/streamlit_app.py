import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Inject Custom CSS to match your original 'style_trial.css'
st.markdown("""
    <style>
    /* 1. Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 2. Main Container (The White Card) */
    .main .block-container {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 600px;
    }

    /* 3. Typography */
    h1 {
        color: #333;
        text-align: center;
        font-size: 28px;
        margin-bottom: 10px;
        font-weight: 700;
    }
    p {
        color: #666;
        font-size: 14px;
    }
    
    /* 4. Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        color: white;
        border-color: transparent;
    }
    
    /* 5. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. OPENAI SETUP (READY FOR API KEY) ---
# This checks for the key. If missing, it uses the Mock Client automatically.
api_key = os.environ.get("OPENAI_API_KEY")

class MockResponse:
    def __init__(self):
        class Message:
            content = "Mocked suggestion: Maintain a balanced diet rich in whole grains and vegetables. Exercise regularly for at least 30 minutes a day. (Add API Key to enable real AI advice)"
        class Choice:
            message = Message()
        self.choices = [Choice()]

class MockClient:
    def __init__(self, api_key=None): pass
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs): return MockResponse()

if api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except:
        client = MockClient()
else:
    client = MockClient()

def get_ai_suggestions(is_diabetic):
    try:
        if is_diabetic:
            prompt = "Provide 3-4 brief, actionable lifestyle and health management suggestions for someone predicted to have diabetes."
        else:
            prompt = "Provide 3-4 brief, actionable suggestions for maintaining good health and preventing diabetes."
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful health advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unable to generate suggestions at this time."

# --- 3. LOAD MODEL & SCALER ---
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    
    # Load Model
    if os.path.exists('model.pkl'):
        model = pickle.load(open('model.pkl', 'rb'))
    
    # Load Scaler (Matches your training columns: Glucose, Insulin, BMI, Age)
    try:
        if os.path.exists('diabetes.csv'):
            dataset = pd.read_csv('diabetes.csv')
            X = dataset.iloc[:, [1, 4, 5, 7]].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X)
        else:
            # Fallback scaler if CSV is missing
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.array([[0,0,0,0], [200, 800, 60, 100]])) 
    except:
        pass
        
    return model, scaler

model, scaler = load_resources()

# --- 4. APP LAYOUT ---
st.title("Diabetes Risk Predictor")
st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Enter your health metrics to assess your diabetes risk</p>", unsafe_allow_html=True)

# Input Form
with st.form("predictor_form"):
    # Using 'help' to replace the 'i' info tooltip from your original HTML
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100, step=1, help="Normal fasting: 70-100 mg/dL")
    st.caption("Low: 0-70 | Normal: 70-100 | High: 100+")
    
    insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, max_value=900.0, value=30.0, step=0.1, help="Normal range: 2-20 μU/mL")
    st.caption("Low: 0-2 | Normal: 2-20 | High: 20+")
    
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="BMI = weight(kg) / height(m)²")
    st.caption("Underweight: <18.5 | Normal: 18.5-25 | Overweight: 25+")
    
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
    
    st.markdown("---")
    submitted = st.form_submit_button("Analyze Risk")

# --- 5. RESULTS ---
if submitted:
    if model and scaler:
        # Predict
        input_data = np.array([[glucose, insulin, bmi, age]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        # Calculate Likelihood
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(scaled_data)[0][int(prediction[0])]
            likelihood_score = round(prob * 100, 1)
        else:
            likelihood_score = 0
            
        is_diabetic = (prediction[0] == 1)
        
        # Tabs Logic (Matching your original structure)
        st.markdown("### Results")
        tab1, tab2, tab3 = st.tabs(["Prediction", "Likelihood", "Diet Plan"])
        
        with tab1:
            if is_diabetic:
                st.error("You have Diabetes, please consult a Doctor.")
            else:
                st.success("You don't have Diabetes.")
                
        with tab2:
            st.info(f"Probability: {likelihood_score}%")
            # Progress bar matching your visual style
            st.progress(likelihood_score / 100)
            
        with tab3:
            with st.spinner("Generating diet plan..."):
                suggestion = get_ai_suggestions(is_diabetic)
                st.write(suggestion)
                
    else:
        st.error("Error: Model could not be loaded. Please check your repository files.")