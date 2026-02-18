import streamlit as st
from forms.form_variables import *
import pandas as pd 

@st.dialog("Saved Successfully ✅")
def save_popup():
    st.success("Your data has been saved.")

@st.dialog("All data cleared successfully 🧹")
def clear_data_dialog():
    st.success("Your data has been cleared.")

    if st.button("OK"):
        
        st.session_state["demographics"] = None
        st.session_state["diabetes"] = None
        st.session_state["kidney"] = None
        st.session_state["ckc_dietary_details"] = None

        st.rerun()

@st.dialog("Prediction Results 🧹")
def prediction_success():
    st.success("Prediction completed. Please check the results.")


def view_demographics_data(df0):
        
    df= df0.copy()
    df1=pd.DataFrame()
    df1['Patient Name'] = df['name']
    df1['Age'] = df['Age']
    df1['Gender'] = df['Gender'].map(reverse_gender_options)
    df1['Ethnicity'] = df['Ethnicity'].map(reverse_ethnicity_options)
    df1['EducationLevel'] = df['EducationLevel'].map(reverse_education_level_options)
    df1['SocioeconomicStatus'] = df['SocioeconomicStatus'].map(reverse_socioeconomicStatus_options)
    df1['EmploymentStatus'] = df['EmploymentStatus'].map(reverseemploymentStatus_options)
    df1['Smoking'] = df['Smoking'].map({1:'Yes',0:'No'})
    df1['AlcoholConsumption'] = df['AlcoholConsumption'].map(star_symbol_map)
    df1['DietQuality'] = df['DietQuality'].map(star_symbol_map)
    df1['SleepQuality'] = df['SleepQuality'].map(star_symbol_map)    
    df1['BMI- kg/m²'] = df['bmi'].round(2)
    df1['Height-cm'] = df['Height_cm'].round(2)
    df1['Weight-kg'] = df['Weight_kg'].round(2)

    
    return df1.T.reset_index()


def view_diabetes_data(df0):
    
    df= df0.copy()
    df1=pd.DataFrame()
    
    #Lifestyle & Risk History

    df1['Cardiovascular Disease History']=df['cardiovascular_history'].map({1:'Yes',0:'No'})
    df1['Family History of Diabetes']=df['family_history_diabetes'].map({1:'Yes',0:'No'})
    df1['History of Hypertension']=df['hypertension_history'].map({1:'Yes',0:'No'})
    df1['Smoking_status'] = df['Smoking_status'].map(reverse_smoking_options)
    
    #Body Measurements & Vitals

    df1['Systolic BP (mmHg)']=df['systolic_bp']
    df1['Diastolic BP (mmHg)']=df['diastolic_bp']
    df1['Heart Rate (bpm)']=df['heart_rate']
    df1['Waist to Hip Ratio']=df['waist_to_hip_ratio']
    
    #Lipid Profile
    df1['Total Cholesterol (mg/dL)']=df['cholesterol_total']
    df1['HDL Cholesterol (mg/dL)']=df['hdl_cholesterol']
    df1['LDL Cholesterol (mg/dL)']=df['ldl_cholesterol']
    df1['Triglycerides (mg/dL)']=df['triglycerides']

    #Glucose & Metabolic Parameters
    
    df1['Fasting Glucose (mg/dL)']=df['glucose_fasting']
    df1['Fasting Insulin (µIU/mL)']=df['insulin_level']
    df1['Postprandial Glucose (mg/dL)']=df['glucose_postprandial']
    df1['HbA1c (%)']=df['hba1c']

    return df1.T.reset_index()

def view_kidney_data(df0):
    
    df= df0.copy()
    df1=pd.DataFrame() 
    
    #Medical History
    df1['Family History Kidney Disease']=df['FamilyHistoryKidneyDisease'].map({1:'Yes',0:'No'}) 
    df1['Previous Acute Kidney  Injury']=df['PreviousAcuteKidneyInjury'].map({1:'Yes',0:'No'}) 
    df1['Urinary Tract Infections']=df['UrinaryTractInfections'].map({1:'Yes',0:'No'}) \
    
    #Blood Pressure

    df1['Systolic BP (mmHg)']=df['SystolicBP']
    df1['Diastolic BP (mmHg)']=df['DiastolicBP']

    #Blood Sugar

    df1['Fasting Blood Sugar (mg/dL)']=df['FastingBloodSugar']
    df1['HbA1c (%)']=df['HbA1c']

    # Kidney Function Tests

    df1['Serum Creatinine (mg/dL)']=df['SerumCreatinine']
    df1['Blood Urea Nitrogen (mg/dL)']=df['BUNLevels']
    df1['eGFR (mL/min/1.73m²)']=df['GFR']

    #Urine Analysis
    df1['Protein in Urine']=df['ProteinInUrine'].map({1:'Present',0:'Absent'})
    df1['Albumin-Creatinine Ratio (ACR)']=df['ACR']

    #Serum Electrolytes
    df1['Serum Sodium (mEq/L)']=df['SerumElectrolytesSodium']
    df1['Serum Potassium (mEq/L)']=df['SerumElectrolytesPotassium']
    df1['Serum Calcium (mg/dL)']=df['SerumElectrolytesCalcium']
    df1['Serum Phosphorus (mg/dL)']=df['SerumElectrolytesPhosphorus']

    #Hematology
    df1['Hemoglobin (g/dL)']=df['HemoglobinLevels']

    #Medications
    df1['AcE Inhibitors Use']=df['ACEInhibitors'].map({1:'Yes',0:'No'})
    df1['Diuretes Use']=df['Diuretics'].map({1:'Yes',0:'No'})
    df1['Anti Diabetic Medications Use']=df['AntidiabeticMedications'].map({1:'Yes',0:'No'}) 
    df1['NSAIDs Use (times/week)']=df['NSAIDsUse']

    #Symptoms

    df1['Edema']=df['Edema'].map({1:'Yes',0:'No'})
    df1['Fatigue Level']=df['FatigueLevels'].map(star_symbol_map)
    df1['Muscle Cramps']=df['MuscleCramps'].map(star_symbol_map)
    df1['Itching']=df['Itching'].map(star_symbol_map)

    #Monitoring Frequency
    df1['Medical Checkups Frequency (per year)']=df['MedicalCheckupsFrequency']

    return df1.T.reset_index()

def view_ckc_dietary_data(df0):
    
    df= df0.copy()
    df1=pd.DataFrame(df)  
    df1.rename(columns={'Family_History_CRC': 'Family_History_CRC'}, inplace=True)

    #Lifestyle & Dietary Habits
    df1['Pre-existing-Conditions']=df['Pre-existing-Conditions'].map(reverse_pre_existing_conditions_options)
    df1['Family History of  Cancer']=df['Family_History_CRC'].map({1:'Yes',0:'No'})
    df1['Lifestyle']=df['Lifestyle'].map(reverselifestyle_options)


    return df1.T.reset_index()

def view_other_data(df0):
    
    df= df0.copy()
    df1=pd.DataFrame()  

    return df.T.reset_index()
