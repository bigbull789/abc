# Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

train.info()
train.head()

# Correlation heatmap (numeric columns only)
train_numeric = train.select_dtypes(include=['int64', 'float64'])
train_numeric.corr().style.background_gradient(cmap='BuGn')

# Drop Unnecessary Columns
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# Check Missing Values
train.isna().sum()
test.isna().sum()

#Fill Missing Values

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].dropna().mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].dropna().mean())


#Feature Engineering (Sex & Age)

guess_ages = np.zeros((2,3))
combine = [train, test]

# Convert Sex to numeric
for ds in combine:
    ds['Sex'] = ds['Sex'].map({'female':1, 'male':0}).astype(int)

# Fill missing Age
for ds in combine:
    for i in range(2):
        for j in range(3):
            guess_df = ds[(ds['Sex']==i) & (ds['Pclass']==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
    for i in range(2):
        for j in range(3):
            ds.loc[(ds['Age'].isnull()) & (ds['Sex']==i) & (ds['Pclass']==j+1), 'Age'] = guess_ages[i,j]
    ds['Age'] = ds['Age'].astype(int)



# Cleaned Dataset Preview (Sir’s Format)

train_cleaned = train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
train_cleaned.head()

#Prepare Data for Modeling

X_train = pd.get_dummies(train.drop(['Survived'], axis=1))
X_test = pd.get_dummies(test)
y_train = train['Survived']

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


#  Function to Print Scores
# ------------------------------
def print_scores(model, X_train, y_train, predictions, cv_splits=10):
    print("Training Accuracy: %.5f" % model.score(X_train, y_train))
    CV_scores = cross_val_score(model, X_train, y_train, cv=cv_splits)
    print("Cross-validation scores:\n", CV_scores)
    print("Minimum CV score: %.3f" % min(CV_scores))
    print("Maximum CV score: %.3f" % max(CV_scores))
    print("Mean CV score: %.5f ± %0.2f" % (CV_scores.mean(), CV_scores.std()*2))


# 11. Train Random Forest Classifier
# ------------------------------
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=5,
    max_features=8,
    min_samples_split=3,
    random_state=7
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)


# 12. Evaluate Model
# ------------------------------
print_scores(model, X_train, y_train, predictions)
