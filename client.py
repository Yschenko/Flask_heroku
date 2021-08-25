import numpy as np
import pandas as pd
import requests


def main():
    X_test = pd.read_csv('X_test.csv')
    y_preds = np.loadtxt('preds.csv')
    data = X_test.sample(1)
    index = data.index[0]
    dict_to_req = data.to_dict(orient='records')[0]

    pred = requests.get('http://127.0.0.1:4444/predict_churn', params=dict_to_req).text
    print(f'answer from server: {pred}')
    print(f'true ansewr: {int(y_preds[index])}')


if __name__ == '__main__':
    main()