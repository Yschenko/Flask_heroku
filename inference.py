import numpy as np
import pandas as pd
import pickle
from joblib import load
from flask import Flask, request
import os

MODEL = load('churn_model.pkl')
app = Flask(__name__)


@app.route('/predict_churn')
def exp_number():

    return str(int(MODEL.predict(pd.DataFrame(request.args, index=[0]))))


def main():
    port = 4444 # int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()