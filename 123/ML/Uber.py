import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("/content/uber.csv")

df = df.dropna()  # remove missing values
df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0)]

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

df = df[df['fare_amount'] < df['fare_amount'].quantile(0.99)]

columns_to_drop = ['Unnamed: 0', 'key', 'pickup_datetime']
df_corr = df.drop(columns=columns_to_drop, errors='ignore')
print("\nCorrelation Matrix:\n", df_corr.corr()['fare_amount'])
#df_corr.style.background_gradient(cmap='BuGn')

X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)

def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> R²: {r2:.3f}, RMSE: {rmse:.3f}")


evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, rf_pred, "Random Forest")


plt.scatter(y_test, rf_pred, alpha=0.3)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Random Forest Predictions")
plt.show()