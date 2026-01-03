import streamlit as st
from methods.custom_funtions import save_popup
from forms.form_variables import ethnicity_options,gender_options,education_level_options,socioeconomicStatus_options,employmentStatus_options
def demographics_form():


    with st.form("demographics_form"):

        st.subheader("🩺 Patient Demographics Fields Form ")

        col1, col2 = st.columns(2)

        with col1:

            # colum 1 starts here
            name = st.text_input("Name")
            
            age = st.number_input(
            "Age",
            min_value=0,
            max_value=90,
            value=15
        )

            ethnicity_label = st.radio(
            "Ethnicity",
            list(ethnicity_options.keys()),
            horizontal=False
        )

            ethnicity = ethnicity_options[ethnicity_label]

            employment_status_lablel = st.radio(
            "Employment",
            list(employmentStatus_options.keys()),
            horizontal=True
        )   
            employment_status = employmentStatus_options[employment_status_lablel]
            
            smoking_label = st.toggle("Smoking")

            smoking = 1 if smoking_label else 0

            alcohol_consumption = st.slider(
            "Alcohol Consumption Rating (0 = No ,1 = Very Low, 10 = Very High)",
            min_value=0,
            max_value=10,
            value=1    )
            
            # colum 1 ends here
        
        # colum 2 starts here
        with col2:

            gender_label = st.selectbox(
            "Gender",
            list(gender_options.keys())
            
        )
            gender = gender_options[gender_label]

            education_level_label = st.radio(
            "Education",
            list(education_level_options.keys()),
            horizontal=True
        )   
            education_level = education_level_options[education_level_label]

            income_level_label = st.radio(
            "Income",
            list(socioeconomicStatus_options.keys()),
            horizontal=True
        )
            income_level = socioeconomicStatus_options[income_level_label]

            height_value=st.number_input("Height - in cm",min_value=100,max_value=250,value=150)
            weight_value=st.number_input("Weight - in kg",min_value=30,max_value=150,value=70)

            bmi_value=st.number_input("BMI - kg/m²",min_value=15,max_value=50,value=24)
            
            # colum 2 starts here

        #common pane starts here

        
        


        diet_score = st.slider(
            "Diet Score (1 = Very Poor, 10 = Very Good)",
            min_value=1,
            max_value=10,
            value=5
        )

        sleep_score = st.slider(
            "Sleep Quality (1 = Very Poor, 10 = Very Good)",
            min_value=1,
            max_value=10,
            value=5
        )

        demographics_details_submit = st.form_submit_button("Save")

        if demographics_details_submit:
                        
            st.session_state["demographics"] = {
            "name": name,
            "Age": age,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "EducationLevel": education_level,
            "SocioeconomicStatus": income_level,
            "EmploymentStatus": employment_status,
            "Smoking":smoking,
            "AlcoholConsumption": alcohol_consumption,
            "SleepQuality": sleep_score,
            "bmi": bmi_value,
            "DietQuality": diet_score,
            "Height_cm": height_value,
            "Weight_kg": weight_value
        }
            
            save_popup()
    
