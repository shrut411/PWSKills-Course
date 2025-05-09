import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Load sample dataset (Iris for Q2)
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target']
df = df.drop('target', axis=1)

# Artificially add some missing values and categorical column
df.loc[0:5, 'sepal length (cm)'] = np.nan
df['category'] = pd.cut(df['sepal width (cm)'], bins=3, labels=['low', 'medium', 'high'])

# Q1: Define numeric and categorical features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = ['category']

# Split features and target
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Automated feature selection
selector_model = RandomForestClassifier(random_state=42)
selector = SelectFromModel(selector_model, threshold='mean')

# Numerical pipeline
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Full pipeline with feature selection and classifier
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('feature_selection', selector),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Q1 Random Forest Pipeline Accuracy:", accuracy_score(y_test, y_pred))

# Q2: Voting classifier pipeline on Iris dataset
X2 = iris.data
y2 = iris.target
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Voting classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(max_iter=200))
], voting='hard')

voting_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('voter', voting_clf)
])

voting_pipeline.fit(X2_train, y2_train)
y2_pred = voting_pipeline.predict(X2_test)
print("Q2 Voting Classifier Accuracy on Iris:", accuracy_score(y2_test, y2_pred))
