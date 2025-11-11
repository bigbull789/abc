import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("/content/Churn_Modelling.csv")

df

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

df

df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

df

df.value_counts("Geography")

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

df

X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(32,16),
                      activation='relu',
                      solver='adam',
                      early_stopping=True,
                      random_state=42,
                      max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))