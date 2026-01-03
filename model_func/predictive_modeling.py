import joblib
import pandas as pd
import streamlit as st
import os
import glob
#import xgboost
from ml_models import *

from forms.form_variables import *

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)   # your saved model

def model_predict(df1, df2, path, diseases_name):

    result_df = pd.DataFrame()

    # Concatenate input data
    raw_data = pd.concat([df1, df2], axis=1)

    # Find model files
    model_files = glob.glob(path)

    if not model_files:
        raise FileNotFoundError(f"No model files found in path: {path}")

    for model_path in model_files:

        # Load model
        model = joblib.load(model_path)

        # Required columns from model
        cols = model.feature_names_in_

        # Ensure all columns exist
        missing_cols = set(cols) - set(raw_data.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")

        # Prediction
        prediction = model.predict(raw_data[cols])

        # Clean model name (portable)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        result_df[model_name] = prediction

    # Insert disease name
    result_df.insert(0, "Disease", diseases_name)

    # Mean of predictions
    result_df[diseases_name] = (
        result_df.select_dtypes(include="number").mean(axis=1)
    )

    return result_df


def view_demographics_data(df):
        
    df['Gender'] = df['Gender'].map(reverse_gender_options)
    df['Ethnicity'] = df['Ethnicity'].map(reverse_ethnicity_options)
    df['EducationLevel'] = df['EducationLevel'].map(reverse_education_level_options)
    df['SocioeconomicStatus'] = df['SocioeconomicStatus'].map(reverse_socioeconomicStatus_options)
    df['EmploymentStatus'] = df['EmploymentStatus'].map(reverseemploymentStatus_options)
    
    dis1=df.T.reset_index() 
    dis1.columns = ["Demographics Parameter", "Patient Value"]
    dis1.index=dis1.index+1
    st.dataframe(dis1)


def view_diabetes_data(df):
        
    df['Smoking_status'] = df['Smoking_status'].map(reverse_smoking_options)

    dis2=df.T.reset_index() 
    dis2.columns = ["Diabetic Parameter", "Patient Value"]
    dis2.index=dis2.index+1
    st.dataframe(dis2)