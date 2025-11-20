import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Ensure directories exist
os.makedirs('model', exist_ok=True)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/Heart_attack_dataset.csv', skiprows=1)

# 2. Feature Engineering & Target Creation (Synthetic Logic based on Domain Knowledge)
# Normalizing biomarkers for score calculation
df['cTnI_norm'] = (df['cTnI_ng_mL'] - df['cTnI_ng_mL'].min()) / (df['cTnI_ng_mL'].max() - df['cTnI_ng_mL'].min())
df['BNP_norm'] = (df['BNP_pg_mL'] - df['BNP_pg_mL'].min()) / (df['BNP_pg_mL'].max() - df['BNP_pg_mL'].min())
df['Myo_norm'] = (df['Myoglobin_ng_mL'] - df['Myoglobin_ng_mL'].min()) / (df['Myoglobin_ng_mL'].max() - df['Myoglobin_ng_mL'].min())

# Synthetic Risk Score: Troponin I is the strongest predictor (0.5 weight), BNP (0.3), Myoglobin (0.2)
df['Risk_Score'] = 0.5 * df['cTnI_norm'] + 0.3 * df['BNP_norm'] + 0.2 * df['Myo_norm']

# Define Risk Categories
# Using percentiles to ensure balanced classes for this demo
low_thresh = df['Risk_Score'].quantile(0.33)
high_thresh = df['Risk_Score'].quantile(0.66)

def categorize_risk(score):
    if score > high_thresh:
        return 'High'
    elif score > low_thresh:
        return 'Moderate'
    else:
        return 'Low'

df['Risk_Category'] = df['Risk_Score'].apply(categorize_risk)

# Drop intermediate calculation columns
df = df.drop(columns=['cTnI_norm', 'BNP_norm', 'Myo_norm', 'Risk_Score'])

print(f"Risk Distribution:\n{df['Risk_Category'].value_counts()}")

# 3. Prepare for Modeling
X = df.drop(columns=['Sample_ID', 'Risk_Category'])
y = df['Risk_Category']

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Save Label Encoder mapping for API
# 0: High, 1: Low, 2: Moderate (Order depends on alphabetical sort usually)
# Let's explicitly check classes
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Model Training & Tuning
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(verbose=-1),
    'SVM': SVC(probability=True)
}

best_model = None
best_score = 0
best_name = ""

print("\nTraining Models...")

# Scaling is crucial for SVM and LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

for name, model in models.items():
    # Simple cross-validation or training
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: Accuracy = {acc:.4f}")
    
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

print(f"\nBest Model: {best_name} with Accuracy: {best_score:.4f}")

# 5. Retrain Best Model on Full Training Data with Hyperparameter Tuning (Simplified for script)
# We will use the best model type found.

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', best_model)
])

final_pipeline.fit(X_train, y_train) # Pipeline handles scaling

# Verify
final_acc = final_pipeline.score(X_test, y_test)
print(f"Final Pipeline Accuracy: {final_acc:.4f}")

# 6. Save Artifacts
joblib.dump(final_pipeline, 'model/pipeline.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
# Saving scaler separately just in case, but it is in pipeline
joblib.dump(scaler, 'model/scaler.pkl') 

# Save feature names for API validation
joblib.dump(X.columns.tolist(), 'model/features.pkl')

print("Artifacts saved in model/ directory.")

