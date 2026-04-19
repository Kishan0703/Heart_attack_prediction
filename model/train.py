import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import shap

# Ensure directories exist
os.makedirs('model', exist_ok=True)

def categorize_risk(score, low_thresh, high_thresh):
    """
    Categorizes the risk score into Low, Moderate, or High.
    
    Args:
        score: The synthetic risk score.
        low_thresh: Threshold for Low risk.
        high_thresh: Threshold for High risk.
        
    Returns:
        String category ('Low', 'Moderate', or 'High').
    """
    if score > high_thresh:
        return 'High'
    elif score > low_thresh:
        return 'Moderate'
    else:
        return 'Low'

def train_and_save():
    """
    Main training pipeline: loads data, performs feature engineering,
    selects the best model using Stratified K-Fold CV, and saves artifacts.
    """
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
    low_thresh = df['Risk_Score'].quantile(0.33)
    high_thresh = df['Risk_Score'].quantile(0.66)

    df['Risk_Category'] = df['Risk_Score'].apply(lambda x: categorize_risk(x, low_thresh, high_thresh))

    # Drop intermediate calculation columns
    df = df.drop(columns=['cTnI_norm', 'BNP_norm', 'Myo_norm', 'Risk_Score'])

    print(f"Risk Distribution:\n{df['Risk_Category'].value_counts()}")

    # 3. Prepare for Modeling
    X = df.drop(columns=['Sample_ID', 'Risk_Category'])
    y = df['Risk_Category']

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 4. Model Training & Tuning
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""

    print("\nTraining Models with 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            if isinstance(X_train, pd.DataFrame):
                X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            else:
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]
                
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            cv_scores.append(accuracy_score(y_val, preds))
            
        avg_acc = np.mean(cv_scores)
        print(f"{name}: CV Accuracy = {avg_acc:.4f}")
        
        if avg_acc > best_score:
            best_score = avg_acc
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name} with CV Accuracy: {best_score:.4f}")

    # 5. Retrain Best Model on Full Training Data
    final_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', best_model)
    ])
    final_pipeline.fit(X_train, y_train)

    # Verify
    final_acc = final_pipeline.score(X_test, y_test)
    print(f"Final Pipeline Accuracy: {final_acc:.4f}")

    # 6. Compute and Save SHAP Explainer
    print("Computing SHAP explainer...")
    clf = final_pipeline.named_steps['classifier']
    scaler = final_pipeline.named_steps['scaler']
    X_train_transformed = scaler.transform(X_train)

    try:
        if isinstance(clf, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
            explainer = shap.TreeExplainer(clf)
        else:
            background = shap.sample(X_train_transformed, 100)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
        joblib.dump(explainer, 'model/shap_explainer.pkl')
        print("SHAP explainer saved to model/shap_explainer.pkl")
    except Exception as e:
        print(f"Could not compute SHAP explainer: {e}")

    # 7. Save Artifacts
    joblib.dump(final_pipeline, 'model/pipeline.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')
    joblib.dump(scaler, 'model/scaler.pkl') 
    joblib.dump(X.columns.tolist(), 'model/features.pkl')

    print("Artifacts saved in model/ directory.")

if __name__ == "__main__":
    train_and_save()
