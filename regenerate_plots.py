import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import shap

# Settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 100

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "ckm0.96666666.xlsx")
model_dir = os.path.join(base_dir, "models")
plot_dir = os.path.join(base_dir, "plots")

print("Loading data...")
# Load Data
df = pd.read_excel(data_path)
X = df.drop('Result', axis=1)
y = df['Result']

# Split (Replicate original split)
print("Splitting data...")
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Scale
print("Scaling data...")
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_train_scaled = pd.DataFrame(scaler.transform(X_train_full), columns=X_train_full.columns)

# Feature names with units
feature_names_with_units = [
    'AGE [years]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
    'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
]

# Keep original column names for model prediction compatibility
# X_test_scaled.columns = feature_names_with_units
# X_train_scaled.columns = feature_names_with_units

# Models
model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]

for model_file in model_files:
    model_name = model_file.replace('_model.pkl', '')
    print(f"Processing {model_name}...")
    try:
        model = joblib.load(os.path.join(model_dir, model_file))
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue
    
    # 1. ROC Curve (Test Set)
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test_scaled)
            # Normalize to 0-1 for ROC if needed, or just use decision function scores
        else:
            y_prob = model.predict(X_test_scaled) # Fallback
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        save_path = os.path.join(plot_dir, "ROC_Curves", f"roc_curve_{model_name}_test.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        # 2. Recall Curve (Test Set)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        save_path = os.path.join(plot_dir, "Recall_Curves", f"recall_curve_{model_name}_test.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error generating ROC/Recall for {model_name}: {e}")

    # 3. SHAP Summary (Training Set)
    try:
        # Subsample for speed
        X_background = X_train_scaled.sample(n=min(100, len(X_train_scaled)), random_state=42)
        
        explainer = None
        shap_values = None
        
        # TreeExplainer
        if model_name in ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'ExtraTrees', 'DecisionTree', 'GradientBoosting']:
             try:
                 explainer = shap.TreeExplainer(model)
                 shap_values = explainer(X_background)
                 shap_values.feature_names = feature_names_with_units
             except Exception as e:
                 print(f"TreeExplainer failed for {model_name}: {e}")
        
        # Fallback to KernelExplainer
        if shap_values is None:
             # Use a small background summary for KernelExplainer
             kmeans_summary = shap.kmeans(X_train_scaled, 10)
             f = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
             explainer = shap.KernelExplainer(f, kmeans_summary)
             
             # Calculate SHAP values
             sv = explainer.shap_values(X_background)
             
             # Handle list output (for classification)
             if isinstance(sv, list):
                 sv = sv[1] # Positive class
             
             # Create Explanation object
             shap_values = shap.Explanation(values=sv, base_values=explainer.expected_value, data=X_background.values, feature_names=feature_names_with_units)
             # Note: passed X_background.values to avoid column name conflict if any, though Explanation handles it.
             
             if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                 shap_values.base_values = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, show=False)
        # plt.title(f"SHAP Summary - {model_name}", fontsize=14) 
        plt.tight_layout()
        save_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{model_name}_shap_summary.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        # Importance Plot (Bar)
        plt.figure()
        shap.summary_plot(shap_values, plot_type="bar", show=False)
        plt.title(f"Feature Importance - {model_name}", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{model_name}_shap_importance.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
         print(f"Error generating SHAP for {model_name}: {e}")

print("Done regenerating plots.")
