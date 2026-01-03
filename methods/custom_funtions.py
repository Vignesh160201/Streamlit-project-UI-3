import streamlit as st

@st.dialog("Saved Successfully ✅")
def save_popup():
    st.success("Your data has been saved.")

@st.dialog("All data cleared successfully 🧹")
def clear_data_dialog():
    st.success("Your data has been cleared.")

    if st.button("OK"):
        
        st.session_state["demographics"] = None
        st.session_state["diabetes"] = None
        st.session_state["kidney"] = None
        st.session_state["ckc_dietary_details"] = None

        st.rerun()





