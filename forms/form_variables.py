
import os

star_symbol_map = { 0: "", 1: "⭐", 2: "⭐", 3: "⭐⭐", 4: "⭐⭐", 5: "⭐⭐⭐", 6: "⭐⭐⭐", 7: "⭐⭐⭐⭐", 8: "⭐⭐⭐⭐", 9: "⭐⭐⭐⭐⭐", 10: "⭐⭐⭐⭐⭐" }

gender_options ={
    'Male': 0,
    'Female': 1, 
    'Other': 2
}

reverse_gender_options = {v: k for k, v in gender_options.items()}

ethnicity_options = {
    'White / Caucasian': 0,
    'Black / African American': 1,
    'Asian': 2,
    'Hispanic/others': 3
}

reverse_ethnicity_options = {v: k for k, v in ethnicity_options.items()}

education_level_options = {
    'None/No formal': 0,
    'High School': 1,
    'Bachelors/Graduate': 2,
    'Higher/Postgraduate': 3}

reverse_education_level_options = {v: k for k, v in education_level_options.items()}

socioeconomicStatus_options = {
    'Lower-Middle': 0,
    'Low':0,
    'Middle': 1,
    'Upper-Middle': 1,
    'High': 2  
        }

reverse_socioeconomicStatus_options = {v: k for k, v in socioeconomicStatus_options.items()}

smoking_options = {
        "Never": 0,
        "Former": 1,
        "Current": 2
    }

reverse_smoking_options = {v: k for k, v in smoking_options.items()}

employmentStatus_options={'Employed':0,'Unemployed':1,'Retired':2,'Student':3}

reverseemploymentStatus_options = {v: k for k, v in employmentStatus_options.items()}

lifestyle_options = {
    'Sedentary': 0,
    'Smoker': 1,
    'Moderate Exercise': 2,
    'Active': 3
}
reverselifestyle_options = {v: k for k, v in lifestyle_options.items()}

family_history_options = {'Yes': 1,'No': 0}

reversefamily_history_options = {v: k for k, v in family_history_options.items()}

pre_existing_conditions_options = {
    'None': 0,
    'Diabetes': 1,
    'Obesity': 2,
    'Hypertension': 3
}

reverse_pre_existing_conditions_options = {v: k for k, v in pre_existing_conditions_options.items()}

diet_options={
    'Unhealthy':0,
    'Average':1,
    'Healthy':2
}

reverse_diet_options = {v: k for k, v in diet_options.items()}

#'ATA' 'NAP' 'ASY' 'TA'

chestpaintype_options={
    'Atypical Angina - ATA':0,
    'Non-Anginal Pain -NAP':1,
    'Asymptomatic - ASY':2,
    'Typical Angina - TA':3
}
reverse_chestpaintype_options = {v: k for k, v in chestpaintype_options.items()}

#RestingECG :['Normal' 'ST' 'LVH']
restingecg_options={
    'Normal':0,
    'ST-T Wave Abnormality':1,
    'Left Ventricular Hypertrophy':2
}

reverse_restingecg_options = {v: k for k, v in restingecg_options.items()}

#ST_Slope :['Up' 'Flat' 'Down']

st_slope_options={
    'Upsloping':0,
    'Flat':1,
    'Downsloping':2
}

reverse_st_slope_options = {v: k for k, v in st_slope_options.items()}

BASE_DIR = os.getcwd()  # or __file__ logic

diabetes_model_path = os.path.join(
        BASE_DIR, "ml_models", "diabetes_models", "*.pkl")

kidney_model_path =os.path.join(
        BASE_DIR, "ml_models", "kidney_models", "*.pkl")

colorectal_cancer_model_path =os.path.join(
        BASE_DIR, "ml_models", "colorectal_cancer_models", "*.pkl")

heart_disease_model_path =os.path.join(
        BASE_DIR, "ml_models", "heart_models", "*.pkl")

stroke_model_path =os.path.join(
        BASE_DIR, "ml_models", "stroke_models", "*.pkl")

#diabetes_model_path="/workspaces/Streamlit-project-UI-2/ml_models/diabetes_models/*.pkl"
#kidney_model_path="/workspaces/Streamlit-project-UI-2/ml_models/kidney_models/*.pkl"


