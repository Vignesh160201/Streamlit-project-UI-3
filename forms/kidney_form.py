import streamlit as st
from methods.custom_funtions import save_popup


def kidney_form():

    with st.form("kidney_form"):

        st.subheader("🩺 Patiney Kidney Field Form")

        # -------------------------------
        # 1. Medical History
        # -------------------------------
        st.markdown("### 📋 Medical History")

        col1, col2, col3 = st.columns(3)

        with col1:
            family_history = st.selectbox(
                "Family History of Kidney Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col2:
            previous_aki = st.selectbox(
                "Previous Acute Kidney Injury",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            uti_history = st.selectbox(
                "Recurrent Urinary Tract Infections",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        # -------------------------------
        # 2. Blood Pressure
        # -------------------------------
        st.markdown("### 🩸 Blood Pressure")

        col1, col2 = st.columns(2)

        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)

        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)

        # -------------------------------
        # 3. Blood Sugar
        # -------------------------------
        st.markdown("### 🍬 Blood Sugar")

        col1, col2 = st.columns(2)

        with col1:
            fasting_blood_sugar = st.number_input(
                "Fasting Blood Sugar (mg/dL)", 50, 400, 100
            )

        with col2:
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5, step=0.1)

        # -------------------------------
        # 4. Kidney Function
        # -------------------------------
        st.markdown("### 🧪 Kidney Function Tests")

        col1, col2, col3 = st.columns(3)

        with col1:
            serum_creatinine = st.number_input(
                "Serum Creatinine (mg/dL)", 0.1, 15.0, 1.0, step=0.1
            )

        with col2:
            bun = st.number_input("BUN Levels (mg/dL)", 1, 200, 20)

        with col3:
            gfr = st.number_input("GFR (mL/min/1.73m²)", 1, 150, 90)

        # -------------------------------
        # 5. Urine Tests
        # -------------------------------
        st.markdown("### 🚽 Urine Analysis")

        col1, col2 = st.columns(2)

        with col1:
            protein_urine = st.selectbox(
                "Protein in Urine",
                [0, 1],
                format_func=lambda x: "Present" if x == 1 else "Absent"
            )

        with col2:
            acr = st.number_input("Albumin-Creatinine Ratio (ACR)", 0, 5000, 30)

        # -------------------------------
        # 6. Electrolytes
        # -------------------------------
        st.markdown("### ⚖️ Serum Electrolytes")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sodium = st.number_input("Sodium (mEq/L)", 120, 170, 140)

        with col2:
            potassium = st.number_input(
                "Potassium (mEq/L)", 2.0, 8.0, 4.0, step=0.1
            )

        with col3:
            calcium = st.number_input(
                "Calcium (mg/dL)", 5.0, 15.0, 9.5, step=0.1
            )

        with col4:
            phosphorus = st.number_input(
                "Phosphorus (mg/dL)", 1.0, 10.0, 3.5, step=0.1
            )

        # -------------------------------
        # 7. Hematology
        # -------------------------------
        st.markdown("### 🧬 Hematology")

        hemoglobin = st.number_input(
            "Hemoglobin Levels (g/dL)", 3.0, 20.0, 13.5, step=0.1
        )

        # -------------------------------
        # 8. Medications
        # -------------------------------
        st.markdown("### 💊 Medications")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            ace_inhibitors = st.toggle("ACE Inhibitors")

        with col2:
            diuretics = st.toggle("Diuretics")

        with col3:
            antidiabetic = st.toggle("Antidiabetic Medications")

        with col4:
            nsaids_use = st.slider("NSAIDs Use (times/week)", 0, 10, 0)

        # -------------------------------
        # 9. Symptoms
        # -------------------------------
        st.markdown("### ⚠️ Symptoms")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            edema = st.selectbox("Edema", [0, 1])

        with col2:
            fatigue = st.slider("Fatigue Level", 0, 10, 3)

        with col3:
            muscle_cramps = st.slider("Muscle Cramps", 0, 10, 2)

        with col4:
            itching = st.slider("Itching", 0, 10, 1)

        # -------------------------------
        # 10. Monitoring
        # -------------------------------
        st.markdown("### 📆 Monitoring")

        medical_checkups = st.slider(
            "Medical Checkups Frequency (per year)", 0, 12, 2
        )

        # -------------------------------
        # Submit Button
        # -------------------------------
        submit_kidney = st.form_submit_button("Save")

        if submit_kidney:

            st.session_state["kidney"] = {
                "FamilyHistoryKidneyDisease": family_history,
                "PreviousAcuteKidneyInjury": previous_aki,
                "UrinaryTractInfections": uti_history,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "FastingBloodSugar": fasting_blood_sugar,
                "HbA1c": hba1c,
                "SerumCreatinine": serum_creatinine,
                "BUNLevels": bun,
                "GFR": gfr,
                "ProteinInUrine": protein_urine,
                "ACR": acr,
                "SerumElectrolytesSodium": sodium,
                "SerumElectrolytesPotassium": potassium,
                "SerumElectrolytesCalcium": calcium,
                "SerumElectrolytesPhosphorus": phosphorus,
                "HemoglobinLevels": hemoglobin,
                "ACEInhibitors": int(ace_inhibitors),
                "Diuretics": int(diuretics),
                "NSAIDsUse": nsaids_use,
                "AntidiabeticMedications": int(antidiabetic),
                "Edema": edema,
                "FatigueLevels": fatigue,
                "MuscleCramps": muscle_cramps,
                "Itching": itching,
                "MedicalCheckupsFrequency": medical_checkups
            }

            save_popup()

    
