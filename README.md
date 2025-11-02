# Student Exam Score Prediction

Un projet de prédiction des notes d'examen des étudiants en utilisant le Machine Learning (`scikit-learn`) et MLflow pour le suivi des métriques.

---

## 🎯 Objectif

Prédire la note d’examen (`exam_score`) d’un étudiant à partir de :  

- `hours_studied` : Nombre d'heures d'étude  
- `sleep_hours` : Heures de sommeil  
- `attendance_percent` : Pourcentage de présence  
- `previous_scores` : Notes précédentes  

Le projet compare plusieurs modèles :

- `LinearRegression`  
- `Ridge`  
- `Lasso`  
- `VotingRegressor`  

Les métriques sont suivies avec **MLflow** et le modèle final est sauvegardé avec **pickle**.

1. Cloner le dépôt :

```bash
git clone https://github.com/Sitylist94/student_exam_score.git
cd student_exam_score
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Télécharger le dataset depuis Kaggle et placer le fichier student_exam_scores.csv dans le dossier data/ :

Dataset Kaggle : https://www.kaggle.com/datasets/grandmaster07/student-exam-score-dataset-analysis

## 🏃‍♂️ Exécution du script

```bash
python src/model.py
```


Le script va :

- `Entraîner les modèles sur le dataset.`

- `Afficher les scores d’entraînement et de test.`

- `Sauvegarder le modèle final dans models/model.pkl.`

- `Logguer les métriques avec MLflow.`


## Pour lancer MLflow :

```bash
mlflow ui
```

## 🔍 Tester des prédictions sur de nouvelles données

```python
import numpy as np

# Exemple : [hours_studied, sleep_hours, attendance_percent, previous_scores]

sample = np.array([[2, 9, 90, 85]])
prediction = model_4.predict(sample)
print("Predicted exam score:", prediction[0])
```

## 📝 Notes importantes

- Le fichier .pkl n’est pas inclus dans le dépôt pour des raisons de taille et confidentialité.

- Le dataset doit être téléchargé depuis Kaggle.

- Les colonnes des nouvelles données doivent être dans le même ordre que pour l'entraînement.


## 🔧 Librairies utilisées

- `numpy`

- `pandas`

- `scikit-learn`

- `matplotlib`

- `mlflow`
