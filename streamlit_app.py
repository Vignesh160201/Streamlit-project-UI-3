import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import lightgbm

from forms.demographic_form import demographics_form
from forms.diabetes_form import diabetes_form
from forms.kidney_form import kidney_form
from forms.ckc_dietarty_details_form import ckc_dietary_details_form
from forms.other_details_form import other_details_form

from methods.custom_funtions import clear_data_dialog
from model_func.predictive_modeling import *
from forms.form_variables import diabetes_model_path,kidney_model_path,colorectal_cancer_model_path,heart_disease_model_path

st.set_page_config(
    page_title="Health Prediction System",
    layout="wide"
)

st.title("🎈Chronic Diseases Prediction Based on Patient Lifestyle Details")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Demographics Details", "Diabetes Details",
                                      "Kidney  Details","Colorectal Cancer Dietary details",
                                      "Others  Details","Detailed View & Predictions"])


with tab1:
    demographics_form()
    demographics_data=st.session_state.get("demographics") 

with tab2:
    diabetes_form()
    diabetes_data=st.session_state.get("diabetes")
    
with tab3:
    kidney_form()
    kidney_data=st.session_state.get("kidney")

with tab4:
    ckc_dietary_details_form()
    ckc_data=st.session_state.get("ckc_dietary_details")

with tab5:
    other_details_form()
    other_details_data=st.session_state.get("other_details")

with tab6:
    #st.json(demographics_data)
    #st.json(diabetes_data)
    #st.json(kidney_data)
    #st.json(ckc_data)
    #st.json(other_details_data)

    if diabetes_data or demographics_data or kidney_data or ckc_data or other_details_data:

        demographics_df=pd.DataFrame([demographics_data])
        diabetes_df = pd.DataFrame([diabetes_data])
        kidney_df=pd.DataFrame([kidney_data])
        ckc_dietary_df=pd.DataFrame([ckc_data])
        other_df=pd.DataFrame([other_details_data])

        heart_df=pd.concat([diabetes_df, other_df], axis=1)

        #view_demographics_data(demographics_df)
        dis1=demographics_df.T.reset_index() 
        dis1.columns = ["Demographics Parameter", "Patient Value"]
        dis1.index=dis1.index+1
        st.dataframe(dis1)
        
        #view_diabetes_data(diabetes_df)
        dis2=diabetes_df.T.reset_index() 
        dis2.columns = ["Diabetic Parameter", "Patient Value"]
        dis2.index=dis2.index+1
        st.write(dis2)

        dis3=kidney_df.T.reset_index() 
        dis3.columns = ["Kidney Parameter", "Patient Value"]
        dis3.index=dis3.index+1
        st.write(dis3)

        dis4=ckc_dietary_df.T.reset_index() 
        dis4.columns = ["Colorectal Cancer Dietary Parameter", "Patient Value"]
        dis4.index=dis4.index+1
        st.write(dis4)

        dis5=other_df.T.reset_index() 
        dis5.columns = ["Other Health Parameter", "Patient Value"]
        dis5.index=dis5.index+1
        st.write(dis5)


    else:
        st.warning("Data not available")


    if st.button("Predict"):

        if(demographics_data is None or diabetes_data is None or kidney_data is None or ckc_data is None or other_details_data is None):
            
            st.error("Please fill 2 or more forms atleast before making predictions.")
        else:
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
            label_col = df.columns   # Feature
            value_col = df.iloc[0]# Binary Value (0/1)

            labels = df.columns.tolist()
            values = df.iloc[0].tolist()

            sizes = [1] * len(values)

            colors = ['green' if v == 0 else 'blue' for v in values]

            # Plot
            fig, ax = plt.subplots()

            ax.pie(
                sizes,
                labels=[f"{l}\n{v}" for l, v in zip(labels, values)],
                colors=colors,
                startangle=90,
                wedgeprops={'width': 0.4, 'edgecolor': 'black'}
            )

            ax.set_title("Binary Donut Visualization")
            ax.axis('equal')

            #Streamlit display
            st.pyplot(fig)
            
            st.success("Prediction completed. Please check the results above.")

    if st.checkbox("Accuracy Details"):
        st.dataframe(diabetes_preditions)
        st.dataframe(kidney_predictions)
        st.dataframe(colorectal_cancer_predictions)
        st.dataframe(heart_disease_predictions)

    if st.button("🧹 Clear All Data"):
        clear_data_dialog()
                


    
    
    

    
