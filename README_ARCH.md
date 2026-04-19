# Cardiac Risk Prediction System Architecture

```mermaid
graph TD
    User((User)) -->|Biomarker Data| Frontend[Web Dashboard - index.html]
    Frontend -->|POST /predict| API[FastAPI Server - backend/main.py]
    API -->|X-API-Key Validation| Auth{Auth}
    Auth -->|Valid| Predict[Prediction Service]
    Predict -->|Pipeline & LabelEncoder| MLModel[ML Model Pipeline.pkl]
    Predict -->|UUID| Cache[(Prediction Cache)]
    API -->|GET /explain/id| Explain[Explanation Service]
    Explain -->|SHAP Explainer| SHAP[SHAP Explainer.pkl]
    Explain -->|Retrieve Data| Cache
    
    Data[(CSV Dataset)] -->|model/train.py| Train[Training Pipeline]
    Train -->|Stratified K-Fold| Select[Model Selection]
    Select -->|Best Model| Artifacts[Model Artifacts]
    Artifacts -->|pipeline.pkl| Predict
    Artifacts -->|shap_explainer.pkl| Explain
```

## System Components

1.  **Frontend**: Static HTML5 dashboard using Chart.js for visualization and real-time monitoring simulations.
2.  **Backend**: FastAPI server providing high-performance endpoints for prediction and explanation.
3.  **ML Model**: Scikit-learn/Imblearn pipeline including `StandardScaler`, `SMOTE`, and a Classifier (e.g., Logistic Regression).
4.  **SHAP Explainer**: Post-hoc model explanation using SHAP values to identify feature importance for specific predictions.
5.  **DevOps**: Dockerized deployment with GitHub Actions for CI/CD.
