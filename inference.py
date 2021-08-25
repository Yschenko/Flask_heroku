import numpy as np
import pandas as pd
import pickle
from joblib import load
from flask import Flask, request

MODEL = load('churn_model.pkl')
app = Flask(__name__)

@app.route('/predict_churn')
def exp_number():

    data = {'is_male': int(request.args.get('is_male')),\
            'num_inters': int(request.args.get('num_inters')),\
            'late_on_payment': int(request.args.get('late_on_payment')),\
            'age': int(request.args.get('age')),
            'years_in_contract': float(request.args.get('years_in_contract'))}

    to_predict = pd.DataFrame(data, index=[0])
    return str(int(MODEL.predict(to_predict)))


def main():
    app.run(host='127.0.0.1', port=4444)


if __name__ == '__main__':
    main()