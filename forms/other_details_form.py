import streamlit as st
from methods.custom_funtions import save_popup
from forms.form_variables import  chestpaintype_options,st_slope_options,restingecg_options

#ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope

def other_details_form():

    with st.form("other_details_form"):

        st.subheader("❤️ Other Details Form ")
        st.write("Please fill in the following other details to for checking heart condition:")

        col1, col2 = st.columns(2)

        with col1:

            
            chestpaintype_label = st.selectbox(
                "Chest Pain Type",
                list(chestpaintype_options.keys()),
                help="Select the type of chest pain experienced."
            )
            chestpaintype = chestpaintype_options[chestpaintype_label]  

            restingecg_label = st.selectbox(
                "Resting ECG",
                list(restingecg_options.keys()),
                help="Select the type of resting electrocardiographic results."
            )
            restingecg = restingecg_options[restingecg_label]      
            
            st_slope_label = st.selectbox(
                "ST Slope",
                list(st_slope_options.keys()),
                help="Select the slope of the ST segment during peak exercise."
            )
            st_slope = st_slope_options[st_slope_label] 

            fasting_bs_options = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes']) 
            fasting_bs = 1 if fasting_bs_options == 'Yes' else 0

            exerciseangina = st.toggle(
                "Chest pain when exertion or exercise",
                help="Yes means the person does get chest pain when exercising."
                "No means they do not get chest pain with exercise.."
            )


        with col2:

            resting_bp = st.slider('Resting Blood Pressure (mm Hg) - Normal Range 120/80 mm Hg', 80, 200, 120)
            cholesterol = st.slider('Cholesterol (mg/dl) -  Normal Range - 100 mg/dL ', 60, 600, 100)
            max_hr = st.slider('Maximum Heart Rate Achieved', 60, 210, 150)
            oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.0, 1.0, 0.1)

        other_details_submitted = st.form_submit_button("Save")

        if other_details_submitted:
       

            st.session_state["other_details"] = {
                "ChestPainType": chestpaintype,
                "RestingBP": resting_bp,
                "RestingECG":restingecg,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "MaxHR": max_hr,
                "ExerciseAngina": 1 if exerciseangina else 0,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope
            }                

            save_popup()
    

        

        