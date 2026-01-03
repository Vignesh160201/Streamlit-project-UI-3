import streamlit as st
from methods.custom_funtions import save_popup
from forms.form_variables import smoking_options

def diabetes_form():

    

    with st.form("diabetes_form"):

        st.subheader("🩺 Patient Diabetes Fields Form")

        st.subheader("Lifestyle & Risk History")
        col1, col2 = st.columns(2)

        with col1:
            cardiovascular_history = st.toggle("Cardiovascular Disease History")
            smoking_status_label = st.radio("Smoking Status", list(smoking_options.keys()),horizontal=True)
            smoking_status = smoking_options[smoking_status_label]
            

        with col2:
            family_history_diabetes = st.toggle("Family History of Diabetes")
            hypertension_history = st.toggle("History of Hypertension")
            

        st.divider()
        st.subheader("Body Measurements & Vitals")

        col3, col4 = st.columns(2)

        with col3:

            systolic_bp = st.number_input(
                "Systolic BP (mmHg)",
                min_value=80,
                max_value=200,
                value=120,
                help="Normal: 90–120 mmHg"
            )
            diastolic_bp = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40,
                max_value=130,
                value=80,
                help="Normal: 60–80 mmHg"
            )

        with col4:
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=40,
                max_value=200,
                value=72,
                help="Normal adult resting HR: 60–100 bpm"
            )
            waist_to_hip_ratio = st.number_input(
                "Waist to Hip Ratio",
                min_value=0.50,
                max_value=2.00,
                value=0.90,
                help="Normal: ≤0.90 (Men), ≤0.85 (Women)"
            )

        st.divider()
        st.subheader("Lipid Profile")

        col5, col6 = st.columns(2)

        with col5:
            cholesterol_total = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=100,
                max_value=400,
                value=180,
                help="Desirable: <200 mg/dL"
            )
            hdl_cholesterol = st.number_input(
                "HDL Cholesterol (mg/dL)",
                min_value=10,
                max_value=100,
                value=45,
                help="Normal: ≥40 (Men), ≥50 (Women)"
            )

        with col6:
            ldl_cholesterol = st.number_input(
                "LDL Cholesterol (mg/dL)",
                min_value=10,
                max_value=300,
                value=90,
                help="Optimal: <100 mg/dL"
            )
            triglycerides = st.number_input(
                "Triglycerides (mg/dL)",
                min_value=30,
                max_value=500,
                value=120,
                help="Normal: <150 mg/dL"
            )

        st.divider()
        st.subheader("Glucose & Metabolic Parameters")

        col7, col8 = st.columns(2)

        with col7:
            glucose_fasting = st.number_input(
                "Fasting Glucose (mg/dL)",
                min_value=50,
                max_value=300,
                value=99,
                help="Normal: 70–99 mg/dL"
            )
            insulin_level = st.number_input(
                "Insulin Level (µIU/mL)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                help="Normal: 2–25 µIU/mL"
            )

        with col8:
            glucose_postprandial = st.number_input(
                "Postprandial Glucose (mg/dL)",
                min_value=70,
                max_value=400,
                value=120,
                help="Normal: <140 mg/dL"
            )
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.0,
                max_value=15.0,
                value=5.4,
                help="Normal: <5.7%"
            )

        submit_diabetes = st.form_submit_button("Save")

    if submit_diabetes:
        
        st.session_state["diabetes"] = {
            "Smoking_status" : smoking_status,
            "family_history_diabetes": int(family_history_diabetes),
            "hypertension_history": int(hypertension_history),
            "cardiovascular_history": int(cardiovascular_history),
            "waist_to_hip_ratio": waist_to_hip_ratio,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "cholesterol_total": cholesterol_total,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "triglycerides": triglycerides,
            "glucose_fasting": glucose_fasting,
            "glucose_postprandial": glucose_postprandial,
            "insulin_level": insulin_level,
            "hba1c": hba1c
        }
        save_popup()

          

    
