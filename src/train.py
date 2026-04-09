import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    print("Executing offline batch training pipeline...")
