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
        print(cols)
        print(raw_data[cols])
        print(model_path)
        
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


def predict_button_method(
            demographics_df,diabetes_df,kidney_df,ckc_dietary_df,other_df,heart_df):
    
    diabetes_preditions=model_predict(demographics_df,diabetes_df,diabetes_model_path,"Diabetes Prediction Results")
    #st.dataframe(diabetes_preditions)
            
    kidney_predictions=model_predict(demographics_df,kidney_df,kidney_model_path,"Kidney Disease Prediction Results")
    #st.dataframe(kidney_predictions)

    colorectal_cancer_predictions=model_predict(demographics_df,ckc_dietary_df,colorectal_cancer_model_path,
                                                            "Colorectal Cancer Prediction Results")
    #st.dataframe(colorectal_cancer_predictions)

    heart_disease_predictions=model_predict(demographics_df,heart_df,heart_disease_model_path,
                                                        "Heart Disease Prediction Results")
    #st.dataframe(heart_disease_predictions)

    df = pd.concat([
                    diabetes_preditions.iloc[:, [ -1]],
                    kidney_predictions.iloc[:, [-1]],
                    colorectal_cancer_predictions.iloc[:, [-1]],
                    heart_disease_predictions.iloc[:, [-1]]
                ], axis=1)      

    import matplotlib.pyplot as plt

                # Data from DataFrame
                # Automatically pick columns

    labels = df.columns.tolist()
    values = df.iloc[0].tolist()

    sizes = [1] * len(values)

    colors = ['green' if v == 0 else 'blue' for v in values]

    # Plot
    fig, ax = plt.subplots()

    ax.pie( sizes,
             labels=[f"{l}\n{v}" for l, v in zip(labels, values)],
                    colors=colors,
                    startangle=90,
                    wedgeprops={'width': 0.4, 'edgecolor': 'black'}
                )

    ax.set_title("Binary Donut Visualization")
    ax.axis('equal')

    #Streamlit display
    st.pyplot(fig)
    box = st.checkbox("Accuracy Details")

    if box :
       
       if diabetes_preditions is not None and not diabetes_preditions.empty and box:
        st.dataframe(diabetes_preditions)

       else:
        st.warning("⚠️ Diabetes accuracy details not available")

        if kidney_predictions is not None and not kidney_predictions.empty:
            st.dataframe(kidney_predictions)
        else:
            st.warning("⚠️ Kidney accuracy details not available")

        if colorectal_cancer_predictions is not None and not colorectal_cancer_predictions.empty:
            st.dataframe(colorectal_cancer_predictions)
        else:
            st.warning("⚠️ Colorectal cancer accuracy details not available")

        if heart_disease_predictions is not None and not heart_disease_predictions.empty:
            st.dataframe(heart_disease_predictions)
        else:
            st.warning("⚠️ Heart disease accuracy details not available")

    else:
        st.info("Click Predict before check box option for Accuracy Details")

    return diabetes_preditions,kidney_predictions,colorectal_cancer_predictions,heart_disease_predictions





      
                 
                


