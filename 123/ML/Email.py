import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("/content/emails.csv")

df

df['spam'] = LabelEncoder().fit_transform(df['spam'])

# Handle NaN values before splitting X and y, as some columns (e.g., 'Prediction') contained missing values.
df_processed = df.dropna()

# Define the target variable 'y' from the 'Prediction' column
y = df_processed['Prediction']
y = LabelEncoder().fit_transform(y)

# Define the feature set 'X' by dropping non-numeric 'Email No.' and the target 'Prediction'
X = df_processed.drop(columns=['Email No.', 'Prediction'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)


knn_pred = knn.predict(X_test)

svm = SVC(kernel='linear')

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
