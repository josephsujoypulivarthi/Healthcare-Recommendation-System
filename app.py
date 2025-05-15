import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
@st.cache_resource
def load_models():
    # Load the machine learning model
    with open('models/best_advanced_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    
    # Load the neural network model
    nn_model = keras.models.load_model('models/nn_model.h5')
    
    # Load the scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return ml_model, nn_model, scaler

ml_model, nn_model, scaler = load_models()

# Set page title
st.set_page_config(page_title="Healthcare Recommendation System", layout="wide")

# App title and description
st.title("Healthcare Recommendation System")
st.markdown("""
This application uses Machine Learning and Neural Networks to provide personalized 
healthcare recommendations based on patient health data.

Enter the patient's information below to get a heart disease risk assessment and recommendations.
""")

# Create two columns
col1, col2 = st.columns([1, 1])

# Input form
with col1:
    st.subheader("Patient Information")
    
    age = st.slider("Age", 20, 90, 50)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    sex_value = 0 if sex == "Female" else 1
    
    cp = st.selectbox("Chest Pain Type", 
                    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                    index=0)
    cp_value = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True)
    fbs_value = 0 if fbs == "No" else 1
    
    restecg = st.selectbox("Resting ECG Results", 
                        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                        index=0)
    restecg_value = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True)
    exang_value = 0 if exang == "No" else 1
    
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 0.0, 0.1)
    
    slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                      ["Upsloping", "Flat", "Downsloping"],
                      index=0)
    slope_value = ["Upsloping", "Flat", "Downsloping"].index(slope)
    
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
    
    thal = st.selectbox("Thalassemia", 
                     ["Normal", "Fixed Defect", "Reversible Defect"],
                     index=0)
    thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
    thal_value = thal_map[thal]

# Create a dataframe with the input values
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_value],
    'cp': [cp_value],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_value],
    'restecg': [restecg_value],
    'thalach': [thalach],
    'exang': [exang_value],
    'oldpeak': [oldpeak],
    'slope': [slope_value],
    'ca': [ca],
    'thal': [thal_value]
})

# Add engineered features
input_data['age_chol_ratio'] = input_data['age'] / (input_data['chol'] + 1)
input_data['trestbps_thalach_ratio'] = input_data['trestbps'] / (input_data['thalach'] + 1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
with col2:
    st.subheader("Risk Assessment")
    
    if st.button("Generate Recommendations"):
        # ML model prediction
        ml_prediction = ml_model.predict(input_data)[0]
        ml_probability = ml_model.predict_proba(input_data)[0][1]
        
        # Neural Network prediction
        nn_probability = nn_model.predict(input_scaled)[0][0]
        nn_prediction = 1 if nn_probability > 0.5 else 0
        
        # Average probability
        avg_probability = (ml_probability + nn_probability) / 2
        
        # Display results
        st.markdown("### Heart Disease Risk Assessment")
        
        # Create a gauge chart for risk
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create gauge background
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 0.33, 0.1, color='green', alpha=0.6))
        ax.add_patch(plt.Rectangle((0.33, 0), 0.33, 0.1, color='yellow', alpha=0.6))
        ax.add_patch(plt.Rectangle((0.66, 0), 0.34, 0.1, color='red', alpha=0.6))
        
        # Add pointer
        ax.arrow(avg_probability, 0.2, 0, -0.05, head_width=0.03, head_length=0.05, fc='black', ec='black')
        
        # Remove axes
        ax.set_axis_off()
        
        # Add text
        ax.text(0.165, 0.3, "Low Risk", ha='center')
        ax.text(0.5, 0.3, "Moderate Risk", ha='center')
        ax.text(0.83, 0.3, "High Risk", ha='center')
        ax.text(avg_probability, 0.5, f"{avg_probability:.2%}", ha='center', fontsize=16, fontweight='bold')
        
        st.pyplot(fig)
        
        # Show detailed prediction values
        st.markdown(f"**Machine Learning Model Risk: {ml_probability:.2%}**")
        st.markdown(f"**Neural Network Risk: {nn_probability:.2%}**")
        st.markdown(f"**Average Risk: {avg_probability:.2%}**")
        
        # Recommendations based on risk level
        st.markdown("### Personalized Recommendations")
        
        if avg_probability < 0.33:
            st.success("**Low Risk of Heart Disease**")
            st.markdown("""
            #### Recommendations:
            1. **Maintain Current Lifestyle**: Continue your healthy habits.
            2. **Regular Check-ups**: Schedule annual health screenings.
            3. **Stay Active**: Aim for at least 150 minutes of moderate exercise weekly.
            4. **Balanced Diet**: Continue eating heart-healthy foods.
            """)
        elif avg_probability < 0.66:
            st.warning("**Moderate Risk of Heart Disease**")
            st.markdown("""
            #### Recommendations:
            1. **Medical Consultation**: Schedule a follow-up with a cardiologist.
            2. **Lifestyle Adjustment**: Consider these changes:
               - Reduce sodium intake to less than 2300mg daily
               - Increase physical activity to 30 minutes daily
               - Monitor blood pressure regularly
            3. **Stress Management**: Practice relaxation techniques like meditation.
            4. **Sleep Hygiene**: Ensure 7-8 hours of quality sleep.
            """)
        else:
            st.error("**High Risk of Heart Disease**")
                 st.markdown("""
                 #### Recommendations:
                 1. **Urgent Medical Attention**: Consult with a cardiologist as soon as possible.
                 2. **Medication Review**: Discuss potential medications with your doctor.
                 3. **Strict Dietary Changes**:
                    - Limit saturated fats and trans fats
                    - Reduce sodium to less than 1500mg daily
                    - Increase fruit and vegetable intake
                 4. **Regular Monitoring**: Track blood pressure, cholesterol, and blood sugar daily.
                 5. **Supervised Exercise**: Work with a physical therapist on a safe exercise program.
                 """)
             
             # Show factors contributing to risk
             st.markdown("### Key Risk Factors")
             
             # Determine top risk factors
             feature_importance = {
                 'Age': age if age > 55 else 0,
                 'Cholesterol': chol if chol > 200 else 0,
                 'Blood Pressure': trestbps if trestbps > 130 else 0,
                 'Max Heart Rate': (220 - age - thalach) if (220 - age - thalach) > 0 else 0,
                 'ST Depression': oldpeak if oldpeak > 1 else 0,
                 'Number of Vessels': ca * 2 if ca > 0 else 0
             }
             
             # Sort and display top 3 risk factors
             sorted_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
             
             if any(factor[1] > 0 for factor in sorted_factors):
                 for factor, value in sorted_factors:
                     if value > 0:
                         st.markdown(f"- **{factor}**: Contributing to increased risk")
             else:
                 st.markdown("No significant individual risk factors identified.")