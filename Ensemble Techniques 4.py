import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess the dataset
url = 'https://drive.google.com/uc?id=1bGoIE4ZZkG5nyh-fGZAJ7LH0ki3UfmSJ'  # Convert share link to direct download
df = pd.read_csv(url)

# Handle missing values (if any)
df.dropna(inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('target')
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 2: Split the dataset
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Step 5: Feature Importance
importances = clf.feature_importances_
features = X.columns
top_features_idx = np.argsort(importances)[-5:]
plt.figure(figsize=(8,5))
sns.barplot(x=importances[top_features_idx], y=features[top_features_idx])
plt.title('Top 5 Important Features')
plt.show()

# Step 6: Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 7: Report Best Hyperparameters
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)
print("Tuned Model F1 Score:", f1_score(y_test, best_pred))

# Step 8: Decision Boundary Visualization (Top 2 features)
from matplotlib.colors import ListedColormap

top2_idx = importances.argsort()[-2:]
X_vis = X_train.iloc[:, top2_idx].values
y_vis = y_train.values
model_2f = RandomForestClassifier(**grid_search.best_params_)
model_2f.fit(X_vis, y_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model_2f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolor='k', cmap=ListedColormap(('red', 'green')))
plt.xlabel(features[top2_idx[0]])
plt.ylabel(features[top2_idx[1]])
plt.title("Decision Boundary (Top 2 Features)")
plt.show()
