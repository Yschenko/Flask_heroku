import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from joblib import dump


def main():
    df = pd.read_csv('cellular_churn_greece.csv')
    X = df.loc[:, df.columns[df.columns != 'churned']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=26)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'score: {clf.score(X_test, y_test)}')
    print(f'confusion_matrix:\n {confusion_matrix(y_test, y_pred)}')
    print(f'confusion_matrix:\n {classification_report(y_test, y_pred)}')
    model = pickle.dumps(clf, )
    dump(clf, 'churn_model.pkl')
    X_test.to_csv('X_test.csv', index=False)
    np.savetxt('preds.csv', clf.predict(X_test), delimiter=',')


if __name__ == '__main__':
    main()
