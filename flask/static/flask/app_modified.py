import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from openai import OpenAI
import os

app = Flask(__name__)
model = pickle.load(open('flask/static/css/model.pkl', 'rb'))

dataset = pd.read_csv('flask/static/css/diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

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

# Initialize OpenAI client
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))

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
    
diet_val_1 = get_ai_suggestions(1)
print(diet_val_1)

@app.route('/')
def home():
    return render_template('index_modified.html')

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
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )
    likelihood_val = model.predict_proba( sc.transform(final_features) )[0][int(prediction)]

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred
    prediction_val = prediction
    pred_text = pred

    return render_template(
        'index_modified.html',
        prediction_text='{}'.format(output))

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
            'index_modified.html', 
            prediction_text='{}'.format(pred_text),
            likelihood_text='{}'.format(likelihood_val),
            diet_recommend_text='{}'.format((diet_show if diet_flag else "")))
    else:
        return render_template(
            'index_modified.html')


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
            'index_modified.html', 
            prediction_text='{}'.format(pred_text),
            likelihood_text='{}'.format(likelihood_val if likelihood_flag else ""),
            diet_recommend_text='{}'.format(diet_show))
    else:
        return render_template(
            'index_modified.html')

if __name__ == "__main__":
    app.run(debug=True)
