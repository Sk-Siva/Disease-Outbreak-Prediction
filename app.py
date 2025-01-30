import os
import pickle
import streamlit as st    
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

# Construct the correct path for model files
model_dir = os.path.join(os.getcwd(), "saved_models")

diabetes_model_path = os.path.join(model_dir, "diabetes_model.sav")
heart_disease_model_path = os.path.join(model_dir, "heart_model.sav")

# Load models
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))


# Sidebar menu
with st.sidebar:
    selected = option_menu('Prediction of disease outbreak system',
                           ['Diabetes Prediction', 'Heart Disease Prediction', ],
                           menu_icon='hospital-fill', icons=['activity', 'heart',], default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        Bloodpressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input('Age of the person')
    with col2:
        Sex = st.text_input('Gender (1 = Male, 0 = Female)')
    with col3:
        ChestPain = st.text_input('Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)')
    with col1:
        RestingBloodPressure = st.text_input('Resting Blood Pressure')
    with col2:
        Cholesterol = st.text_input('Serum Cholesterol level')
    with col3:
        FastingBloodSugar = st.text_input('Fasting Blood Sugar (1 = > 120 mg/dl, 0 = < 120 mg/dl)')
    with col1:
        RestingECG = st.text_input('Resting Electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)')
    with col2:
        MaxHeartRate = st.text_input('Maximum Heart Rate achieved')
    with col3:
        ExerciseInducedAngina = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1:
        STDepression = st.text_input('Depression induced by exercise relative to rest')
    with col2:
        Slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        MajorVessels = st.text_input('Number of major vessels colored by fluoroscopy')
    with col1:
        Thalassemia = st.text_input('Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect)')

    heart_disease_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [Age, Sex, ChestPain, RestingBloodPressure, Cholesterol, FastingBloodSugar,
                      RestingECG, MaxHeartRate, ExerciseInducedAngina, STDepression, Slope,
                      MajorVessels, Thalassemia]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_disease_diagnosis = 'The person has heart disease'
        else:
            heart_disease_diagnosis = 'The person does not have heart disease'

    st.success(heart_disease_diagnosis)