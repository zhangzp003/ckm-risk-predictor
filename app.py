import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

# 设置页面配置
st.set_page_config(
    page_title="CKM综合征早期（1-2期）风险预测工具",
    page_icon="🏥",
    layout="wide"
)

# 标题和简介
st.title("🏥 CKM综合征早期（1-2期）风险预测工具")
st.markdown("""
本工具基于机器学习模型，通过患者的临床特征预测CKM综合征早期（1-2期）的风险概率。
请输入以下 7 个特征值进行预测。
""")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 侧边栏：模型选择
st.sidebar.header("⚙️ 设置")
# 修改为相对路径
model_dir = os.path.join(current_dir, "models")
data_path = os.path.join(current_dir, "data", "ckm0.96666666.xlsx")
plot_dir = os.path.join(current_dir, "plots")
roc_dir = os.path.join(current_dir, "plots", "ROC_Curves")

# 获取可用模型
try:
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        model_names = [f.replace('_model.pkl', '') for f in model_files]
        
        # 默认选择性能较好的模型（如CatBoost或XGBoost，如果存在），否则默认第一个
        default_index = 0
        preferred_models = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest']
        for pref in preferred_models:
            if pref in model_names:
                default_index = model_names.index(pref)
                break
                
        selected_model_name = st.sidebar.selectbox("选择预测模型", model_names, index=default_index)
    else:
        st.error(f"模型目录不存在: {model_dir}")
        st.stop()
except Exception as e:
    st.error(f"无法读取模型目录: {e}")
    st.stop()

# 加载模型和标准化器
@st.cache_resource
def load_resources(model_name):
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"加载资源失败: {e}")
        return None, None

model, scaler = load_resources(selected_model_name)

# 加载训练数据（用于SHAP背景）
@st.cache_resource
def load_data():
    try:
        df = pd.read_excel(data_path)
        # 假设前7列是特征，最后一列是标签，这里我们需要根据特征名提取
        feature_names = ['AGE', 'BMI', 'FBG', 'HBA1C', 'HDL', 'TG', 'UA']
        X = df[feature_names]
        return X
    except Exception as e:
        return None

X_train = load_data()

if model is None or scaler is None:
    st.error("模型或标准化器加载失败，请检查文件路径。")
    st.stop()

st.sidebar.success(f"已加载模型: {selected_model_name}")

# 输入表单
st.subheader("📝 患者特征输入")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("年龄 (AGE) [岁]", min_value=18.0, max_value=120.0, value=60.0, step=1.0, help="患者的年龄，单位：岁")
    bmi = st.number_input("体重指数 (BMI) [kg/m²]", min_value=10.0, max_value=60.0, value=24.0, step=0.1, help="Body Mass Index，单位：kg m⁻²")
    fbg = st.number_input("空腹血糖 (FBG) [mg/dL]", min_value=1.0, max_value=500.0, value=100.0, step=1.0, help="Fasting Blood Glucose，单位：mg/dL")

with col2:
    hba1c = st.number_input("糖化血红蛋白 (HbA1c) [%]", min_value=3.0, max_value=20.0, value=6.0, step=0.1, help="Hemoglobin A1c，单位：%")
    hdl = st.number_input("高密度脂蛋白 (HDL) [mg/dL]", min_value=1.0, max_value=200.0, value=50.0, step=1.0, help="High-Density Lipoprotein，单位：mg/dL")
    tg = st.number_input("甘油三酯 (TG) [mg/dL]", min_value=1.0, max_value=1000.0, value=150.0, step=1.0, help="Triglycerides，单位：mg/dL")

with col3:
    ua = st.number_input("尿酸 (UA) [mg/dL]", min_value=1.0, max_value=20.0, value=5.0, step=0.1, help="Uric Acid，单位：mg/dL")

# 预测逻辑
if st.button("🚀 开始预测", type="primary"):
    # 构建输入数据
    input_data = pd.DataFrame({
        'AGE': [age],
        'BMI': [bmi],
        'FBG': [fbg],
        'HBA1C': [hba1c],
        'HDL': [hdl],
        'TG': [tg],
        'UA': [ua]
    })
    
    # 标准化
    try:
        input_scaled = scaler.transform(input_data)
        
        # 预测
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0][1] # 获取正类概率
        else:
            # 对于不支持概率的模型（如某些SVM配置），使用predict
            pred = model.predict(input_scaled)[0]
            proba = 1.0 if pred == 1 else 0.0
            st.warning("该模型不支持概率输出，仅显示类别预测结果。")
            
        # 显示结果
        st.divider()
        st.subheader("📊 预测结果")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            risk_percent = proba * 100
            if proba > 0.5:
                st.error(f"**高风险** (High Risk)")
                st.metric("风险概率", f"{risk_percent:.2f}%", delta="Risk")
            else:
                st.success(f"**低风险** (Low Risk)")
                st.metric("风险概率", f"{risk_percent:.2f}%", delta="-Safe", delta_color="normal")
        
        with c2:
            st.write("风险概率可视化:")
            st.progress(proba)
            
            if proba > 0.7:
                st.write("⚠️ **建议**: 患者CKM综合征早期（1-2期）风险**很高**，建议密切随访并采取积极干预措施。")
            elif proba > 0.5:
                st.write("⚠️ **建议**: 患者CKM综合征早期（1-2期）风险**较高**，建议关注相关指标并考虑干预。")
            elif proba > 0.3:
                st.write("ℹ️ **建议**: 患者处于**中等风险**，建议定期检查，保持良好生活习惯。")
            else:
                st.write("✅ **建议**: 患者目前风险**较低**，请继续保持健康生活方式。")
            
            with st.expander("查看输入特征摘要"):
                display_data = input_data.copy()
                display_data.columns = [
                    'AGE [岁]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                    'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                ]
                st.dataframe(display_data)

        st.divider()
        
        # 选项卡布局
        tab1, tab2, tab3, tab4 = st.tabs(["📈 特征贡献分析 (SHAP)", "📊 风险因子重要性", "📉 模型性能 (ROC/Recall)", "📋 训练数据摘要"])
        
        with tab1:
            st.subheader("单样本 SHAP 贡献度分析")
            st.markdown("该图展示了各特征对**本次预测结果**的贡献程度。红色表示增加风险，蓝色表示降低风险。")
            
            if X_train is not None:
                try:
                    with st.spinner('正在计算 SHAP 值，请稍候...'):
                        # 准备背景数据 (取样以加快速度)
                        background = shap.maskers.Independent(X_train, max_samples=100)
                        
                        # 创建解释器
                        # 注意：不同模型需要不同的解释器，这里尝试通用方法
                        explainer = None
                        
                        # 尝试使用 TreeExplainer (针对树模型)
                        tree_models = ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'ExtraTrees', 'DecisionTree', 'GradientBoosting']
                        
                        if selected_model_name in tree_models:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except:
                                # 如果失败（例如sklearn版本兼容性），回退到KernelExplainer
                                pass
                        
                        # 如果不是树模型或TreeExplainer失败，使用KernelExplainer (通用但慢)
                        if explainer is None:
                             # 使用预测函数包装器，确保输入格式正确
                             f = lambda x: model.predict_proba(x)[:, 1]
                             # 使用kmeans聚类减少背景样本数，加快计算
                             X_train_summary = shap.kmeans(X_train, 10)
                             explainer = shap.KernelExplainer(f, X_train_summary)
                        
                        # 计算当前样本的SHAP值
                        # 注意：输入需要是DataFrame且列名匹配
                        shap_values = explainer(input_data)

                        # 更新特征名称以包含单位（用于绘图）
                        shap_values.feature_names = [
                            'AGE [岁]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                            'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                        ]
                        
                        # 绘制瀑布图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # shap.plots.waterfall(shap_values[0], show=False) # 旧版可能不支持
                        # 使用 matplotlib 绘制
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e:
                    st.warning(f"无法生成实时 SHAP 图 ({str(e)})。请参考下方的全局重要性图。")
                    # st.error(str(e)) # 调试用
            else:
                st.warning("无法加载训练数据，无法进行实时 SHAP 分析。")

        with tab2:
            st.subheader("全局特征重要性")
            st.markdown("该图展示了模型在整体训练数据上认为最重要的风险因子。")
            
            # 尝试加载预生成的图片
            summary_plot_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{selected_model_name}_shap_summary.png")
            importance_plot_path = os.path.join(plot_dir, "SHAP_Analysis", "Training_Set", f"{selected_model_name}_shap_importance.png")
            
            if os.path.exists(summary_plot_path):
                st.image(summary_plot_path, caption=f"{selected_model_name} SHAP Summary Plot", use_container_width=True)
            elif os.path.exists(importance_plot_path):
                st.image(importance_plot_path, caption=f"{selected_model_name} Feature Importance", use_container_width=True)
            else:
                st.info("暂无该模型的全局重要性图表。")

        with tab3:
            st.subheader("模型性能评估")
            c_roc, c_recall = st.columns(2)
            
            with c_roc:
                st.markdown("**ROC 曲线**")
                
                roc_path = os.path.join(roc_dir, f"roc_curve_{selected_model_name}_test.png")
                
                # 如果没有test，尝试找train或者通用的
                if not os.path.exists(roc_path):
                     roc_path = os.path.join(roc_dir, f"roc_curve_{selected_model_name}.png")
                
                if os.path.exists(roc_path):
                    st.image(roc_path, caption=f"{selected_model_name} ROC Curve", use_container_width=True)
                else:
                    st.info(f"暂无 ROC 曲线图。 (未找到: {roc_path})")
            
            with c_recall:
                st.markdown("**Precision-Recall 曲线**")
                pr_path = os.path.join(plot_dir, "Recall_Curves", f"recall_curve_{selected_model_name}_test.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption=f"{selected_model_name} Recall Curve", use_container_width=True)
                else:
                    st.info("暂无 Recall 曲线图。")

        with tab4:
            st.subheader("训练数据摘要")
            if X_train is not None:
                display_train = X_train.copy()
                display_train.columns = [
                    'AGE [岁]', 'BMI [kg/m²]', 'FBG [mg/dL]', 
                    'HBA1C [%]', 'HDL [mg/dL]', 'TG [mg/dL]', 'UA [mg/dL]'
                ]
                st.write(display_train.describe())
            else:
                st.warning("训练数据加载失败。")

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")
        st.write("详细错误信息:", str(e))

# 页脚
st.markdown("---")
st.caption("注：本工具仅供临床辅助参考，不能替代医生诊断。 | Developed with Streamlit")