# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load the dataset
df = pd.read_csv("/content/uber.csv")

# Step 2: Drop unwanted columns and missing values
df = df.drop(['Unnamed: 0', 'key'], axis=1)
df = df.dropna()

# Step 3: Convert pickup_datetime column to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Step 4: Extract useful date and time features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['minute'] = df['pickup_datetime'].dt.minute
df['second'] = df['pickup_datetime'].dt.second
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['year'] = df['pickup_datetime'].dt.year

plt.figure(figsize=(20,12))
sns.boxplot(data=df)

# Step 5: Remove invalid coordinates
df = df[
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180))
]

# Step 6: Define function to calculate distance (Haversine formula)
def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth’s radius in km
    return c * r

# Step 7: Apply distance function
df['Distance'] = haversine_distance(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
)

# Step 8: Remove outliers and unrealistic records
df = df[df['fare_amount'] < df['fare_amount'].quantile(0.99)]
df = df[
    (df['Distance'] < 60) &                       # distance less than 60 km
    (df['fare_amount'] > 0) &                     # positive fare
    (df['passenger_count'] > 0) &                 # valid passenger count
    ~((df['fare_amount'] > 100) & (df['Distance'] < 1)) &   # no high fare for short trips
    ~((df['fare_amount'] < 100) & (df['Distance'] > 100))   # no low fare for long trips
]

# Step 9: Check correlation (optional for analysis)
corr = df.corr()
corr.style.background_gradient(cmap='BuGn')

# Step 10: Select features and target variable
x = df[['Distance', 'hour', 'day', 'month', 'minute', 'second', 'day_of_week', 'year']]
y = df['fare_amount']

# Step 11: Scale features (X)
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Step 12: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 13: Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Step 14: Train Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Step 15: Define evaluation function
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> R²: {r2:.3f}, RMSE: {rmse:.3f}")

# Step 16: Evaluate both models
evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, rf_pred, "Random Forest")

# Step 17: Compare actual vs predicted for Random Forest
plt.scatter(y_test, rf_pred, alpha=0.3, color='blue')
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare (Random Forest)")
plt.show()
