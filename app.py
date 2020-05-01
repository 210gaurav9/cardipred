import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('cardiac.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features)
    final_features[0]=final_features[0]*365
    df=final_features.reshape(1,-1)
    

    prediction = model.predict(df)>0.3

    output = np.round(prediction)

    return render_template('index.html', prediction_text='The person is having cardiac desease if 1 or else fine $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)