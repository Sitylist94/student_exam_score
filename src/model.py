from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import pickle

df = pd.read_csv('data/student_exam_scores.csv')

df=df.drop(["student_id"], axis=1)

X  = df.drop(["exam_score"], axis=1)
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_1 = LinearRegression()
model_2 = Ridge()
model_3 = Lasso()

model_4 = VotingRegressor([('LNR', model_1), ('RDG', model_2), ('LS', model_3)])

for model in (model_1, model_2, model_3, model_4):
    model.fit(X_train, y_train)
    print(model.__class__.__name__, model.score(X_test, y_test))

# sample = np.array([[2, 9, 90, 85]])
# prediction = model_4.predict(sample)
# print("Predicted exam score:", prediction[0])

with mlflow.start_run():
    model_4.fit(X_train, y_train)
    train_score = model_4.score(X_train, y_train)
    test_score = model_4.score(X_test, y_test)

    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)
    mlflow.sklearn.log_model(model_4, "voting_model")

    print(f"Train: {train_score:.4f}, Test: {test_score:.4f}")

model_path = 'models/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_4, f)

print(f"Modèle sauvegardé dans : {model_path}")