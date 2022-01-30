from flask import Flask, render_template, request
import os
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_Data2.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))
port = int(os.environ.get('PORT', 5000))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('total_sqft')

        #print(location, bhk, bath, sqft)
        input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input)[0]*1e5
        #prediction = pipe.predict(input)
        print(prediction)
        prediction=np.round(prediction, 2)
        if prediction>0:
            return f"Prediction: Rs. {str(prediction)}"
        else:
            return "Very Low or Unusual Entry Found. Try Entering Different Values."
    except:
        return "Invalid Or Empty Entry Found. Please Try Again."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
