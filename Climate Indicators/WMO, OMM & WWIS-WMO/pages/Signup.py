import streamlit as st
st.set_page_config(layout="wide")
from components.header import show_header
show_header()



# --- Custom CSS ---
st.markdown("""
    <style>
    .custom-form-container {
        background-color: #f0f4fa;
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.07);
        max-width: 400px;
        margin: 2rem auto;
    }
    .custom-form-container h2 {
        color: #2a52be;
        text-align: center;
        margin-bottom: 2rem;
    }
    label {
        color: #2a52be !important;
        font-weight: 600 !important;
    }
    .stTextInput input {
        background-color: #eaf0fb;
        color: #1a1a1a;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #2a52be;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1.2rem;
    }
    .stButton button:hover {
        background-color: #1e3a8a;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Form Layout ---
st.markdown('<div class="custom-form-container">', unsafe_allow_html=True)
st.markdown("<h2>Register</h2>", unsafe_allow_html=True)

with st.form("register_form"):
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    register_btn = st.form_submit_button("Register")

if register_btn:
    if password != confirm_password:
        st.error("Passwords do not match!")
    elif len(username) < 3 or len(password) < 3:
        st.warning("Username and password must be at least 3 characters.")
    else:
        # Save user logic here (e.g., write to file/database)
        st.success("Registration successful! You can now login.")

st.markdown('</div>', unsafe_allow_html=True)
