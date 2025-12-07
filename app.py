import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

# Page configuration
st.set_page_config(
    page_title="Early-Stage (Stage 1-2) CKM Syndrome Risk Prediction Tool",
    page_icon="🏥",
    layout="wide"
)

# Title and Description
st.title("🏥 Early-Stage (Stage 1-2) CKM Syndrome Risk Prediction Tool")
st.markdown("""
This tool uses machine learning models to predict the risk probability of early-stage (Stage 1-2) CKM syndrome based on patient clinical features.
Please enter the following 7 feature values to predict.
""")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sidebar: Model Selection
st.sidebar.header("⚙️ Settings")
# Relative paths
model_dir = os.path.join(current_dir, "models")
data_path = os.path.join(current_dir, "data", "ckm0.96666666.xlsx")
plot_dir = os.path.join(current_dir, "plots")
roc_dir = os.path.join(current_dir, "plots", "ROC_Curves")

# Get available models
try:
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        model_names = [f.replace('_model.pkl', '') for f in model_files]
        
        # Default selection (prioritize better models)
        default_index = 0
        preferred_models = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest']
        for pref in preferred_models:
            if pref in model_names:
                default_index = model_names.index(pref)
                break
                
        selected_model_name = st.sidebar.selectbox("Select Prediction Model", model_names, index=default_index)
    else:
        st.error(f"Model directory not found: {model_dir}")
        st.stop()
except Exception as e:
    st.error(f"Cannot read model directory: {e}")
    st.stop()

# Load model and scaler
@st.cache_resource
def load_resources(model_name):
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None, None

model, scaler = load_resources(selected_model_name)

# Load training data (for SHAP background)
@st.cache_resource
def load_data():
    try:
        df = pd.read_excel(data_path)
        feature_names = ['AGE', 'BMI', 'FBG', 'HBA1C', 'HDL', 'TG', 'UA']
        X = df[feature_names]
        return X
    except Exception as e:
        return None

X_train = load_data()

if model is None or scaler is None:
    st.error("Failed to load model or scaler. Please check file paths.")
    st.stop()

st.sidebar.success(f"Model loaded: {selected_model_name}")

# Input Form
st.subheader("📝 Patient Feature Input")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (AGE) [years]", min_value=18.0, max_value=120.0, value=60.0, step=1.0, help="Patient's age in years")
    bmi = st.number_input("Body Mass Index (BMI) [kg/m²]", min_value=10.0, max_value=60.0, value=24.0, step=0.1, help="Body Mass Index, unit: kg/m²")
    fbg = st.number_input("Fasting Blood Glucose (FBG) [mg/dL]", min_value=1.0, max_value=500.0, value=100.0, step=1.0, help="Fasting Blood Glucose, unit: mg/dL")

with col2:
    hba1c = st.number_input("Hemoglobin A1c (HbA1c) [%]", min_value=3.0, max_value=20.0, value=6.0, step=0.1, help="Hemoglobin A1c, unit: %")
    hdl = st.number_input("High-Density Lipoprotein (HDL) [mg/dL]", min_value=1.0, max_value=200.0, value=50.0, step=1.0, help="High-Density Lipoprotein, unit: mg/dL")
    tg = st.number_input("Triglycerides (TG) [mg/dL]", min_value=1.0, max_value=1000.0, value=150.0, step=1.0, help="Triglycerides, unit: mg/dL")

with col3:
    ua = st.number_input("Uric Acid (UA) [mg/dL]", min_value=1.0, max_value=20.0, value=5.0, step=0.1, help="Uric Acid, unit: mg/dL")

# Prediction Logic
if st.button("🚀 Start Prediction", type="primary"):
    # Construct input data
    input_data = pd.DataFrame({
        'AGE': [age],
        'BMI': [bmi],
        'FBG': [fbg],
        'HBA1C': [hba1c],
        'HDL': [hdl],
        'TG': [tg],
        'UA': [ua]
    })
    
    # Scale input
    try:
        input_scaled = scaler.transform(input_data)
        
        # Predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0][1] # Probability of positive class
        else:
            # For models not supporting probability (e.g. some SVM configs)
            pred = model.predict(input_scaled)[0]
            proba = 1.0 if pred == 1 else 0.0
            st.warning("This model does not support probability output, only class prediction is shown.")
            
        # Display Results
        st.divider()
        st.subheader("📊 Prediction Results")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            risk_percent = proba * 100
            if proba > 0.5:
                st.error(f"**High Risk**")
                st.metric("Risk Probability", f"{risk_percent:.2f}%", delta="Risk")
            else:
                st.success(f"**Low Risk**")
                st.metric("Risk Probability", f"{risk_percent:.2f}%", delta="-Safe", delta_color="normal")
        
        with c2:
            st.write("Risk Probability Visualization:")
            st.progress(proba)
            
            if proba > 0.7:
                st.write("⚠️ **Recommendation**: The patient has a **very high** risk of early-stage (Stage 1-2) CKM syndrome. Close follow-up and active intervention are recommended.")
            elif proba > 0.5:
                st.write("⚠️ **Recommendation**: The patient has a **high** risk. Attention to relevant indicators and intervention consideration are recommended.")
            elif proba > 0.3:
                st.write("ℹ️ **Recommendation**: The patient is at **medium risk**. Regular check-ups and maintaining healthy life habits are recommended.")
            else:
                st.write("✅ **Recommendation**: The patient is currently at **low risk**. Please continue maintaining a healthy lifestyle.")
            
            with st.expander("View Input Feature Summary"):
                display_data = input_data.copy()
                display_data.columns = [
                    'AGE [years]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                    'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                ]
                st.dataframe(display_data)

        st.divider()
        
        # Tabs layout
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Feature Contribution (SHAP)", "📊 Risk Factor Importance", "📉 Model Performance (ROC/Recall)", "📋 Training Data Summary"])
        
        with tab1:
            st.subheader("Individual SHAP Contribution Analysis")
            st.markdown("This plot shows the contribution of each feature to the **current prediction**. Red indicates increased risk, blue indicates decreased risk.")
            
            if X_train is not None:
                try:
                    with st.spinner('Calculating SHAP values, please wait...'):
                        # Prepare background data
                        background = shap.maskers.Independent(X_train, max_samples=100)
                        
                        # Create explainer
                        explainer = None
                        
                        # Try TreeExplainer
                        tree_models = ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'ExtraTrees', 'DecisionTree', 'GradientBoosting']
                        
                        if selected_model_name in tree_models:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except:
                                pass
                        
                        # Fallback to KernelExplainer
                        if explainer is None:
                             f = lambda x: model.predict_proba(x)[:, 1]
                             X_train_summary = shap.kmeans(X_train, 10)
                             explainer = shap.KernelExplainer(f, X_train_summary)
                        
                        # Calculate SHAP values
                        shap_values = explainer(input_data)

                        # Update feature names with units
                        shap_values.feature_names = [
                            'AGE [years]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                            'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                        ]
                        
                        # Plot waterfall
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e:
                    st.warning(f"Unable to generate real-time SHAP plot ({str(e)}). Please refer to the global importance plot below.")
            else:
                st.warning("Unable to load training data for real-time SHAP analysis.")

        with tab2:
            st.subheader("Global Feature Importance")
            st.markdown("This plot shows the risk factors considered most important by the model across the entire training dataset.")
            
            # Try loading pre-generated plots
            summary_plot_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{selected_model_name}_shap_summary.png")
            importance_plot_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{selected_model_name}_shap_importance.png")
            
            if os.path.exists(summary_plot_path):
                st.image(summary_plot_path, caption=f"{selected_model_name} SHAP Summary Plot", use_container_width=True)
            elif os.path.exists(importance_plot_path):
                st.image(importance_plot_path, caption=f"{selected_model_name} Feature Importance", use_container_width=True)
            else:
                st.info("No global importance plot available for this model.")

        with tab3:
            st.subheader("Model Performance Evaluation")
            c_roc, c_recall = st.columns(2)
            
            with c_roc:
                st.markdown("**ROC Curve**")
                
                roc_path = os.path.join(roc_dir, f"roc_curve_{selected_model_name}_test.png")
                
                if not os.path.exists(roc_path):
                     roc_path = os.path.join(roc_dir, f"roc_curve_{selected_model_name}.png")
                
                if os.path.exists(roc_path):
                    st.image(roc_path, caption=f"{selected_model_name} ROC Curve", use_container_width=True)
                else:
                    st.info(f"No ROC Curve available. (Not found: {roc_path})")
            
            with c_recall:
                st.markdown("**Precision-Recall Curve**")
                pr_path = os.path.join(plot_dir, "Recall_Curves", f"recall_curve_{selected_model_name}_test.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption=f"{selected_model_name} Recall Curve", use_container_width=True)
                else:
                    st.info("No Recall Curve available.")

        with tab4:
            st.subheader("Training Data Summary")
            if X_train is not None:
                display_train = X_train.copy()
                display_train.columns = [
                    'AGE [years]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                    'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                ]
                st.write(display_train.describe())
            else:
                st.warning("Failed to load training data.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Detailed error info:", str(e))

# Footer
st.markdown("---")
st.caption("Note: This tool is for clinical auxiliary reference only and cannot replace doctor diagnosis. | Developed with Streamlit")