import streamlit as st
from forms.form_variables import lifestyle_options,pre_existing_conditions_options,family_history_options

from methods.custom_funtions import save_popup

def ckc_dietary_details_form():

    with st.form('colorectal_cancer_dietary_details_form'):

        st.subheader("Colorectal Cancer Dietary and Lifestyle Form")

        col1, col2 = st.columns(2)

        with col1:

            lifestyle_label = st.radio(
                "Lifestyle",
                list(lifestyle_options.keys()),
                horizontal=True
            )
            lifestyle = lifestyle_options[lifestyle_label]
            
            family_history_label = st.radio(
                "Family History of Cancer",
                list(family_history_options.keys()),
                horizontal=True
            )   
            family_history = family_history_options[family_history_label]
            # colum 1 ends here

        with col2:
            
            pre_existing_conditions_label = st.radio(
                "Pre-existing Conditions",
                list(pre_existing_conditions_options.keys())
            )
            pre_existing_conditions = pre_existing_conditions_options[pre_existing_conditions_label]
            # colum 2 ends here
        
        
        col3,col4=st.columns(2)

        with col3:
            carbohydrates = st.number_input(
            "Carbohydrates (g)",
            min_value=0.0,
            max_value=500.0,
            value=130.0,
            step=1.0,
            help="Total carbohydrate intake in grams. Recommended daily intake ≈ 130g."
        )

            proteins = st.number_input(
                "Proteins (g)",
                min_value=0.0,
                max_value=300.0,
                value=50.0,
                step=1.0,
                help="Protein intake in grams. Average adult requirement ≈ 50g/day."
            )

            fats = st.number_input(
                "Fats (g)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
                help="Total fat intake in grams. Recommended range is 20–35 percent of daily calories."
            )
            with col4:

                vitamin_a = st.number_input(
                    "Vitamin A (IU)",
                    min_value=0.0,
                    max_value=20000.0,
                    value=3000.0,
                    step=100.0,
                    help="Vitamin A intake in IU. RDA ≈ 3000 IU for adults."
                )

                vitamin_c = st.number_input(
                    "Vitamin C (mg)",
                    min_value=0.0,
                    max_value=2000.0,
                    value=75.0,
                    step=1.0,
                    help="Vitamin C intake in mg. RDA ≈ 75–90 mg/day."
                )

                iron = st.number_input(
                    "Iron (mg)",
                    min_value=0.0,
                    max_value=45.0,
                    value=18.0,
                    step=0.1,
                    help="Iron intake in mg. RDA ≈ 8 mg (men), 18 mg (women)."
                )

        ckc_submit = st.form_submit_button("Save")

    if ckc_submit:

        st.session_state['ckc_dietary_details'] = {

                'Lifestyle': lifestyle,
                'Family_History_CRC': family_history,
                'Pre-existing-Conditions': pre_existing_conditions,
                'Carbohydrates(g)':carbohydrates,
                'Proteins(g)':proteins,	
                'Fats(g)':fats,
                'Vitamin_A(IU)':vitamin_a,
                'Vitamin_C(mg)':vitamin_c,
                'Iron(mg)':iron

            }
        
        save_popup()
        
