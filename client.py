import numpy as np
import pandas as pd
import requests

NUMBER_OF_PREDICTIONS = 5
REQUEST_URL = 'http://127.0.0.1:4444/predict_churn'


def main():
    X_test = pd.read_csv('X_test.csv')
    y_preds = np.loadtxt('preds.csv')

    data = X_test.sample(NUMBER_OF_PREDICTIONS)
    index = data.index
    preds = [y_preds[x].astype(int) for x in index]

    dict_to_req = data.to_dict(orient='records')
    new_preds = []
    for d in dict_to_req:

        new_preds.append(int(requests.get(REQUEST_URL, params=d).text))

    print(f'answer from server: {new_preds}')
    print(f'true ansewr: {preds}')
    if preds == new_preds:
        print('predictions from origin model and the server are the same!')


if __name__ == '__main__':
    main()