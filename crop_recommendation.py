import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print(df["label"].value_counts())
X_noisy = X_test.copy()
X_noisy = X_noisy + np.random.normal(0, 0.5, X_noisy.shape)
print(model.score(X_noisy, y_test))
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
importances = model.feature_importances_
features = X.columns
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()
sorted_idx = np.argsort(importances)[::-1]
def explain_prediction(index):
    sample = X_test.iloc[index]
    pred = model.predict([sample])[0]
    print(pred)
    for feature, value in sample.items():
        print(feature, value)
    for i in sorted_idx[:3]:
        print(features[i], importances[i])

explain_prediction(0)