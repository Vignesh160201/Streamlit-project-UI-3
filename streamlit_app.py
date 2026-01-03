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

from methods.custom_funtions import *

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

    if demographics_data is not None and diabetes_data is not None and kidney_data is not None and  ckc_data is not None and other_details_data is not None :

        demographics_df=pd.DataFrame([demographics_data])
        diabetes_df = pd.DataFrame([diabetes_data])
        kidney_df=pd.DataFrame([kidney_data])
        ckc_dietary_df=pd.DataFrame([ckc_data])
        other_df=pd.DataFrame([other_details_data])

        heart_df=pd.concat([diabetes_df, other_df], axis=1)

        
        dis1=view_demographics_data(demographics_df) 
        dis1.columns = ["Demographics Features", "Demographics Value"]
        #st.dataframe(dis1)
        
        dis2=view_diabetes_data(diabetes_df)
        dis2.columns = ["Diabetic Features", "Diabetes Value"]
        #st.dataframe(dis2)

        dis3=view_kidney_data(kidney_df)
        dis3.columns = ["Kidney Features", "Kidney Value"]
        #st.dataframe(dis3)

        dis4=view_ckc_dietary_data(ckc_dietary_df) 
        dis4.columns = ["Colorectal Cancer Dietary Features", "Value"]
        #st.dataframe(dis4)

        dis5=view_other_data(other_df) 
        dis5.columns = ["Other Health Features", "Patient Value"]
        #st.dataframe(dis5)

        dis6=pd.concat([dis1,dis2,dis3,dis4,dis5],axis=1)
        st.dataframe(dis6)


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
            st.write(df)           
                    
            st.success("Prediction completed. Please check the results above.")


    if st.checkbox("Prediction Details"):

        if 'diabetes_preditions' in locals():
            st.dataframe(diabetes_preditions)
        else:
            st.warning("⚠️ Diabetes prediction not available")

        if 'kidney_predictions' in locals():
            st.dataframe(kidney_predictions)
        else:
            st.warning("⚠️ Kidney prediction not available")

        if 'colorectal_cancer_predictions' in locals():
            st.dataframe(colorectal_cancer_predictions)
        else:
            st.warning("⚠️ Colorectal cancer prediction not available")

        if 'heart_disease_predictions' in locals():
            st.dataframe(heart_disease_predictions)
        else:
            st.warning("⚠️ Heart disease prediction not available")

    else:

        st.info("Click check box option for Predition Details before predict button")

    if st.button("🧹 Clear All Data"):
        clear_data_dialog()
                


    
    
    

    
