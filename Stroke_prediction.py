import streamlit as st
import pandas as pd
import numpy as np
import os
from pickle import load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(FILE_DIR, 'model', 'standard_scaler.pkl')
MODEL_PATH = os.path.join(FILE_DIR, 'model', 'logreg_model.pkl')

scaler = load(open(SCALER_PATH, 'rb'))
logreg = load(open(MODEL_PATH, 'rb'))

st.header('Stroke Probability Prediction')
st.markdown('#### Find out if you are at high risk for having a stroke.')

with st.form('my_form'):
    col1, col2 = st.columns(2)
    with col1:
        sex = st.radio('What is your gender?', ('Female', 'Male'))
        age = st.number_input('What is your age?', value=30, step=1, min_value=18, max_value=100)
        ms = st.checkbox('I am or have been married')
    with col2:
        rt = st.radio('What is your residence type?', ('Urban', 'Rural'))
        job = st.radio('What is your work type?', ('Government job', 'Private',
                                                   'Self-employed', 'Never worked'))
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input('Input your body weight in kg', value=60.0, step=0.5, format="%0.1f")
        height = st.number_input('Input your height in cm', value=170, step=1)
    with col2:
        smoke = st.radio('What is your smoking status?', ('Never smoked', 'Used to smoke', 'Smoking'))
    col1, col2 = st.columns(2)
    with col1:
        gl = st.number_input('Input your average glucose level in mg/dL',
                             value=90.0, step=1.0, format='%0.1f')
    with col2:
        ht = st.checkbox('I have hypertension')
        hd = st.checkbox('I have a heart disease')
    submitted = st.form_submit_button('Calculate Stroke Probability')

if submitted:
    gender = 1 if sex == 'Female' else 0
    hypertension = int(ht)
    heart_disease = int(hd)
    ever_married = int(ms)
    residence_type = 1 if rt == 'Urban' else 0
    bmi = weight / ((height/100) ** 2)
    formerly_smoked = 1 if smoke == 'Used to smoke' else 0
    never_smoked = 1 if smoke == 'Never smoked' else 0
    smokes = 1 if smoke == 'Smoking' else 0
    govt_job = 1 if job == 'Smoking' else 0
    private = 1 if job == 'Smoking' else 0
    self_emp = 1 if job == 'Smoking' else 0

    ans_df = pd.DataFrame({'gender': [gender], 'age': [age], 'hypertension': [hypertension],
                           'heart_disease': [heart_disease], 'ever_married': [ever_married],
                           'Residence_type': [residence_type], 'avg_glucose_level': [gl],
                           'bmi': [bmi], 'smoking_status_formerly smoked': [formerly_smoked],
                           'smoking_status_never smoked': [never_smoked],
                           'smoking_status_smokes': [smokes], 'work_type_Govt_job': [govt_job],
                           'work_type_Private': [private], 'work_type_Self-employed': [self_emp]})

    ans_df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(ans_df[['age',
                                                                           'avg_glucose_level',
                                                                           'bmi']])
    pr = logreg.predict_proba(ans_df)[0,1]
    col1, col2 = st.columns([1, 3])
    col1.metric('Chance of having a stroke', f'{int(pr * 100)}%')
    if pr >= 0.5:
        col2.write('')
        col2.error('Your chances of having a stroke are quite high. ' 
                   'You should be checked frequently.')
    elif pr >= 0.25:
        col2.write('')
        col2.warning('Your chances of having a stroke are moderate. '
                     'You should monitor your health.')
    else:
        col2.write('')
        col2.success('Your chances of having a stroke are small. '
                     'Keep it up.')
