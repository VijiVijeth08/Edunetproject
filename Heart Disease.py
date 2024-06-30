import pandas as pd

# Load the dataset
file_path = 'Healthcare Assistant\heart.csv'
heart_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(heart_data.head())

from sklearn.preprocessing import StandardScaler

# Check for missing values
missing_values = heart_data.isnull().sum()

# Encode categorical variables if necessary
# Normalize numerical features
scaler = StandardScaler()
heart_data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(heart_data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

# Display the first few rows of the preprocessed dataset
print(heart_data.head(), missing_values)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of the target variable
sns.countplot(x='target', data=heart_data)
plt.title('Distribution of Target Variable')
plt.show()

# Plot correlation matrix
correlation_matrix = heart_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Plot age distribution by target
sns.boxplot(x='target', y='age', data=heart_data)
plt.title('Age Distribution by Target')
plt.show()

from sklearn.model_selection import train_test_split

# Define features and target
X = heart_data.drop(columns=['target'])
y = heart_data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create interaction terms or polynomial features
heart_data['age_chol_interaction'] = heart_data['age'] * heart_data['chol']
heart_data['age_thalach_interaction'] = heart_data['age'] * heart_data['thalach']

# Redefine features and target
X = heart_data.drop(columns=['target'])
y = heart_data['target']
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear']
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

best_params, accuracy_best, precision_best, recall_best, f1_best
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Train a Support Vector Machine model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

accuracy_rf, precision_rf, recall_rf, f1_rf, accuracy_svm, precision_svm, recall_svm, f1_svm

print(f'Accuracy: {accuracy_rf:.2f}')
print(f'Precision: {precision_rf:.2f}')
print(f'Recall: {recall_rf:.2f}')
print(f'F1 Score: {f1_rf:.2f}')




