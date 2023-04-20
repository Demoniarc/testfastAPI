# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import ccxt


# 2. Create the app object
app = FastAPI()
pickle_in = open("model_eth.pkl","rb")
classifier_2=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    cex_x = ccxt.binance().fetch_ohlcv('ETH/USDT', '5m')
    open_price = cex_x[499][1]
    prediction = classifier_2.predict([[open_price]])
    return {
        'open price': open_price,
        'prediction': prediction[0][0]
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload