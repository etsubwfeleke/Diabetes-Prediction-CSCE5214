import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from openai import OpenAI, api_key
import os

app = Flask(__name__)
# --- FIX 1: Add error handling for model loading ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    print("⚠️ Error: model.pkl not found. Make sure the file is in the same folder.")
    model = None

# --- FIX 2: Check for dataset path to prevent crash ---
if os.path.exists('flask/diabetes.csv'):
    dataset = pd.read_csv('flask/diabetes.csv')
elif os.path.exists('diabetes.csv'):
    dataset = pd.read_csv('diabetes.csv')
else:
    # Create dummy data if CSV is missing so app doesn't crash
    print("⚠️ CSV not found. Using dummy data for scaler.")
    data = {'A': [0], 'Glucose': [0], 'BP': [0], 'C': [0], 'D': [0], 'BMI': [0], 'E': [0], 'Age': [0]}
    dataset = pd.DataFrame(data)

if dataset.shape[1] > 7:
    dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values
else:
    dataset_X = np.array([[0,0,0,0]])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

likelihood_val = 0
prediction_val = 0
diet_val_0 = "Graze freely."
diet_val_1 = ""
diet_show = ""
pred_text = ""
prediction_flag = 0
likelihood_flag = 0
diet_flag = 0

api_key = os.environ.get("OPENAI_API_KEY")
#------------------------------------------------------------------
# Mock OpenAI client for testing without an API key
if api_key:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
else:
    print("⚠️ No OPENAI_API_KEY found — using mock OpenAI client.")

    class MockResponse:
        def __init__(self):
            # We need an object structure that matches response.choices[0].message.content
            class Message:
                content = "Mocked diet plan: Eat healthy and exercise. (No API Key)"
            class Choice:
                message = Message()
            self.choices = [Choice()]

    # --- FIX 3: Correct the Mock Structure to match client.chat.completions.create ---
class MockClient:
        def __init__(self, api_key=None):
            pass

        # Fix: Structure must match client.chat.completions.create
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return MockResponse()
client = MockClient()
#------------------------------------------------------------------
# Initialize OpenAI client
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))

def get_ai_suggestions(prediction_result):
    """
    Get personalized health suggestions from OpenAI based on diabetes prediction.
    
    Args:
        prediction_result (int): 0 for no diabetes, 1 for diabetes
    
    Returns:
        str: AI-generated health suggestions
    """
    try:
        # Create appropriate prompt based on prediction
        if prediction_result == 1:
            prompt = (
                "A person has been predicted to have diabetes based on their health metrics. "
                "Provide 3-4 brief, actionable lifestyle and health management suggestions "
                "including diet, exercise, and medical consultation advice. Keep it concise and supportive."
            )
        else:
            prompt = (
                "A person has been predicted to not have diabetes based on their health metrics. "
                "Provide 3-4 brief, actionable suggestions for maintaining good health and "
                "preventing diabetes, including diet, exercise, and lifestyle tips. Keep it concise and positive."
            )
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model, you can use "gpt-4o" for better quality
            messages=[
                {"role": "system", "content": "You are a helpful health advisor providing brief, practical health suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Extract and return the suggestion text
        suggestions = response.choices[0].message.content.strip()
        return suggestions
    
    except Exception as e:
        # Return a fallback message if API call fails
        return f"Unable to generate personalized suggestions at this time. Please consult a healthcare professional for advice."
    
try:
    diet_val_1 = get_ai_suggestions(1)
except:
    diet_val_1 = "Default Diet Plan"

print(diet_val_1)

try:
    diet_val_0 = get_ai_suggestions(0)
except:
    diet_val_0 = "Default Healthy Lifestyle Plan"

print(diet_val_0)

@app.route('/')
def home():
    return render_template('index_trial.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global likelihood_val
    global prediction_val
    global pred_text
    global likelihood_flag
    global diet_flag
    global prediction_flag
    
    likelihood_flag = 0
    diet_flag = 0
    prediction_flag = 1
    
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict( sc.transform(final_features) )
        if hasattr(model, "predict_proba"):
            likelihood_val = model.predict_proba( sc.transform(final_features) )[0][int(prediction[0])]
                # Convert to readable percentage string
            likelihood_val = f"{round(likelihood_val * 100, 2)}%"
        else:
                likelihood_val = "N/A"

        if prediction[0] == 1:
                pred = "High Risk: You may have Diabetes."
        else:
                pred = "Low Risk: You don't have Diabetes."
                
        output = pred
        prediction_val = prediction[0]
        pred_text = pred
            
        return render_template(
            'index_trial.html',
            prediction_text='{}'.format(output))
    
    except Exception as e:
        # This will print the exact error to your terminal so you know what went wrong
        print(f"ERROR during prediction: {e}")
        return render_template('index_trial.html', prediction_text="Error: Could not predict. Check terminal for details.")


@app.route('/likelihood',methods=['POST'])
def likelihood():
    '''
    For adding likelihood on HTML GUI
    '''
    global pred_text
    global prediction_flag
    global likelihood_val
    global likelihood_flag
    global diet_show
    global diet_flag

    if prediction_flag == 1:
        likelihood_flag = 1

        return render_template(
            'index_trial.html', 
            prediction_text='{}'.format(pred_text),
            likelihood_text='{}'.format(likelihood_val),
            diet_recommend_text='{}'.format((diet_show if diet_flag else "")))
    else:
        return render_template(
            'index_trial.html')


@app.route('/diet',methods=['POST'])
def diet():
    '''
    For adding diet on HTML GUI
    '''
    global pred_text
    global prediction_flag
    global prediction_val
    global likelihood_val
    global likelihood_flag
    global diet_show
    global diet_flag

    if prediction_flag == 1:
        diet_flag = 1

        if prediction_val == 1:
            diet_show = diet_val_1
        else:
            diet_show = diet_val_0

        return render_template(
            'index_trial.html', 
            prediction_text='{}'.format(pred_text),
            likelihood_text='{}'.format(likelihood_val if likelihood_flag else ""),
            diet_recommend_text='{}'.format(diet_show))
    else:
        return render_template(
            'index_trial.html')

if __name__ == "__main__":
    app.run(debug=True)
