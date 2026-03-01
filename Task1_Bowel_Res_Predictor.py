import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('Task1_Bowel_Resections_pred.pkl')  # 加载训练好的RF模型

# Define the feature options
GSH_options = {
    0: 'No history of gastrointestinal surgery',  # 无胃肠手术史
    1: 'history of gastrointestinal surgery',  # 有胃肠手术史
}

DB_options = {
    0: 'No-stricturing & No-penetrating',  # 非狭窄非穿透
    1: 'Stricturing',  # 狭窄
    2: 'Penetrating'  # 穿透
}

# Streamlit UI
st.title("Bowel Resections Predictor for CD patient in coming 12 months")

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Gender input
sex = st.sidebar.selectbox("Gender (1=Male, 2=Female):", options=[1, 2], format_func=lambda x: 'Male (1)' if x == 1 else 'Female (2)')  # 性别选择框

# Age input
age = st.sidebar.number_input("Age:", min_value=1, max_value=120, value=21)  # 年龄输入框

# Symptoms to diagnosis input
std = st.sidebar.number_input("Symptoms to diagnosis (Months):", min_value=0, max_value=600, value=6)

# Duration input
dur = st.sidebar.number_input("Total Duration:", min_value=0, max_value=600, value=6)

# Gastrointestinal_Surgery_History input
GSH = st.sidebar.selectbox("Gastrointestinal Surgery History:", options=list(GSH_options.keys()), format_func=lambda x: GSH_options[x])

# Disease Behavior input
DB = st.sidebar.selectbox("Disease Behavior:", options=list(DB_options.keys()), format_func=lambda x: DB_options[x])

# L4 involvement input
l4 = st.sidebar.selectbox("L4 involvement:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Vomitting input
vom = st.sidebar.selectbox("Vomitting:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Another_Organ_Complication_History input
AOCH = st.sidebar.selectbox("Another Organ Complication History:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Abdominal_Complication_History input
ACH = st.sidebar.selectbox("Abdominal Complication History\n(Obstruction, Mass, Infection):", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Total Iron-Binding Capacity (umol/l) input
TIBC = st.sidebar.number_input("TIBC (umol/l):", min_value=0, max_value=600, value=60.3)

# Fibrinogen input
Fib = st.sidebar.number_input("Fibrinogen (g/L):", min_value=0, max_value=100, value=2.61)

# Nutritional Support Therapy input
NST = st.sidebar.selectbox("Nutritional Support Therapy:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Process the input and make a prediction
feature_values = [sex, age, std, dur, GSH, DB, l4, vom, AOCH, ACH, TIBC, Fib, NST]
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测Task 1类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测一年内发生并发症
        advice = (
            f"According to our model, the patient is classified as high risk for bowel resection within 12 months. "
            f"The estimated probability of bowel resection over the next year is {probability:.1f}%."
            "This indicates an increased likelihood of bowel damage progression and/or complications that may require surgical intervention."
            "The prediction is intended for risk stratification and decision support and should be interpreted alongside clinical assessment and objective disease activity measures."
            "We recommend closer surveillance and timely reassessment for stricturing/penetrating behavior, nutritional status, and ongoing inflammatory activity, with consideration of treatment escalation and/or early colorectal surgical input where appropriate. "
            "Urgent evaluation is warranted if the patient develops severe abdominal pain, signs of obstruction, persistent high fever/sepsis concern, or significant gastrointestinal bleeding."
        )  # 如果预测会发生并发症，给出相关建议
    else:  # 如果预测不会
        advice = (
            f"According to our model, the patient is classified as low risk for bowel resection within 12 months."
            f"The estimated probability of no bowel resection over the next year is {probability:.1f}%."
            "This result supports a lower short-term surgical risk under the current clinical context; "
            "however, disease course remains dynamic and risk may change with evolution of inflammatory burden or development of stricturing/penetrating complications."
            "We recommend continuing standard-of-care follow-up with routine monitoring of symptoms, inflammatory markers and periodic imaging/endoscopic assessment as clinically indicated." 
            "Re-evaluation is suggested if there is clinical deterioration, new obstructive features, sustained biomarker elevation, or complications suggestive of progression."
        )

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['No Bowel Res', 'Yes'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot

    st.pyplot(plt)  # 显示图表

