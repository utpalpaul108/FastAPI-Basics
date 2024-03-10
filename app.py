from typing import Union
from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
from BankNote import BankNote

app = FastAPI()

@app.get('/')
def index():
    return {'name': 'Gaurango'}

@app.get('/items/{item_id}')
def read_item(item_id):
    return {'item_id': item_id}

@app.post('/predict')
def predict_banknote(data: BankNote):

    classifier = pickle.load(open('classifier.pkl','rb'))

    feature_names = ['variance', 'skewness', 'curtosis', 'entropy']
    imput_data = pd.DataFrame([[data.variance, data.skewness, data.curtosis, data.entropy]], columns=feature_names)
    prediction = classifier.predict(imput_data)
    if (prediction[0]>0.5):
        prediction = 'Fake note'
    else:
        prediction = 'Real bank note'

    return {'prediction':prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)

# Now run: uvicorn app:app --reload


