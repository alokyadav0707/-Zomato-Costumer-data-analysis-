import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (replace 'zomato_data.csv' with the actual file path)
data = pd.read_csv('zomato_data.csv')

# Example columns: ['CustomerID', 'Age', 'OrderFrequency', 'AverageSpend', 'Satisfaction', 'PreferredCuisine']

# Data overview
print(data.head())

# Basic statistics
print(data.describe())

# Data preprocessing
# Handle missing values (if any)
data = data.fillna(data.median())

# Encoding categorical variables (e.g., 'PreferredCuisine')
data = pd.get_dummies(data, columns=['PreferredCuisine'], drop_first=True)

# Splitting features and target
X = data[['Age', 'OrderFrequency', 'AverageSpend']] + list(data.filter(regex='PreferredCuisine'))
y = data['Satisfaction']  # Assuming satisfaction is the target variable

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization - Example: Age vs Satisfaction
plt.figure(figsize=(10, 6))
sns.boxplot(x='Satisfaction', y='Age', data=data)
plt.title('Age vs Satisfaction')
plt.show()

# Example insight generation
age_satisfaction = data.groupby('Satisfaction')['Age'].mean()
print("Average Age by Satisfaction Level:\n", age_satisfaction)

# Save processed data
processed_file = 'processed_zomato_data.csv'
data.to_csv(processed_file, index=False)
print(f"Processed data saved to {processed_file}")
