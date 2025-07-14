from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate_models(data):
    X = data.drop('target', axis=1)
    y = data['target']
    model = LogisticRegression()
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return {'Logistic Regression': acc}
