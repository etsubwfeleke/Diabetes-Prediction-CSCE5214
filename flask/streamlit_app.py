import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# --- 2. SESSION STATE & CALLBACKS ---
if 'glucose' not in st.session_state: st.session_state.glucose = 100
if 'insulin' not in st.session_state: st.session_state.insulin = 30.0
if 'bmi' not in st.session_state: st.session_state.bmi = 25.0
if 'age' not in st.session_state: st.session_state.age = 30
if 'submitted' not in st.session_state: st.session_state.submitted = False

def clear_form():
    st.session_state.glucose = 100
    st.session_state.insulin = 30.0
    st.session_state.bmi = 25.0
    st.session_state.age = 30
    st.session_state.submitted = False

# --- 3. CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* White Card Container */
    .main .block-container {
        background-color: white;
        padding: 2rem 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        max-width: 500px;
        margin-top: 2rem;
    }

    /* Text Colors */
    h1, h2, h3, p, label, .stMarkdown, .stNumberInput label, .stCaption {
        color: #333333 !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 5px;
        padding: 10px 20px;
        color: #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }

    /* Input Fields */
    .stNumberInput input {
        background-color: #f8f9fa;
        color: #333;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }

    /* Primary Button */
    .stButton > button[kind="primary"] {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 0.6rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        color: white;
    }

    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        width: 100%;
        background-color: #f0f0f0;
        color: #333;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Hide UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Risk Box Styling */
    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---
api_key = os.environ.get("OPENAI_API_KEY")

class MockResponse:
    def __init__(self):
        class Message:
            content = "Mocked suggestion: Maintain a balanced diet rich in whole grains. (Add API Key for real advice)"
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
        prompt = "Provide 3 brief lifestyle tips." if is_diabetic else "Provide 3 brief prevention tips."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except:
        return "Eat healthy and exercise."

@st.cache_resource
def load_resources():
    model = None
    scaler = None
    try:
        if os.path.exists('model.pkl'):
            model = pickle.load(open('model.pkl', 'rb'))
        
        if os.path.exists('diabetes.csv'):
            dataset = pd.read_csv('diabetes.csv')
            X = dataset.iloc[:, [1, 4, 5, 7]].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.array([[0,0,0,0], [200, 800, 60, 100]])) 
    except:
        pass
    return model, scaler

model, scaler = load_resources()

# --- 5. UI LAYOUT ---
st.markdown("<h1>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 25px;'>Enter your health metrics to assess your diabetes risk</p>", unsafe_allow_html=True)

# Inputs
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, key='glucose', help="Normal fasting: 70-100 mg/dL")
st.caption("Low: 0-70 | Normal: 70-100 | High: 100+")

insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, max_value=900.0, step=0.1, key='insulin', help="Normal range: 2-20 μU/mL")
st.caption("Low: 0-2 | Normal: 2-20 | High: 20+")

bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, step=0.1, key='bmi', help="BMI = weight(kg) / height(m)²")
st.caption("Underweight: <18.5 | Normal: 18.5-25 | Overweight: 25+")

age = st.number_input("Age (years)", min_value=1, max_value=120, key='age')

st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

# Buttons
col1, col2 = st.columns([1, 1]) 
if st.button("Analyze Risk", type="primary"):
    st.session_state.submitted = True

st.button("Clear All", type="secondary", on_click=clear_form)

# --- 6. RESULTS SECTION (TABS RESTORED) ---
if st.session_state.submitted:
    st.markdown("---")
    
    if model and scaler:
        # Predict
        input_data = np.array([[st.session_state.glucose, st.session_state.insulin, st.session_state.bmi, st.session_state.age]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        # Calculate Likelihood
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(scaled_data)[0][int(prediction[0])]
            likelihood_score = round(prob * 100, 1)
        else:
            likelihood_score = 0
            
        is_diabetic = (prediction[0] == 1)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Prediction", "Likelihood", "Diet Plan"])

        with tab1:
            if is_diabetic:
                st.markdown(f"""
                    <div class='result-box high-risk'>
                        <h3 style='margin:0; color: inherit;'>High Risk: You may have Diabetes.</h3>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='result-box low-risk'>
                        <h3 style='margin:0; color: inherit;'>Low Risk: You don't have Diabetes.</h3>
                    </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown(f"<h3 style='text-align: center;'>Probability: {likelihood_score}%</h3>", unsafe_allow_html=True)
            st.progress(likelihood_score / 100)
            
        with tab3:
            with st.spinner("Generating diet plan..."):
                advice = get_ai_suggestions(is_diabetic)
                st.info(advice)
            
    else:
        st.error("Model not loaded. Please check repository.")