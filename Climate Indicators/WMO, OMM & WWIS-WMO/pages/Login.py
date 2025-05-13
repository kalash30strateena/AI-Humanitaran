import streamlit as st
st.set_page_config(layout="wide")
from components.header import show_header
show_header()

st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.form_submit_button("Login")

if login_btn:
    # Replace with your authentication logic
    if username == "admin" and password == "admin":
        st.success("Login successful!")
        st.session_state['user'] = username
        st.experimental_rerun()  # Optionally reload or redirect
    else:
        st.error("Invalid username or password.")
