import streamlit as st
import psycopg2
import bcrypt
import datetime

# --- Streamlit Page Config and Styling ---
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
        }
        h1:hover a, h2:hover a, h3:hover a, h4:hover a, h5:hover a, h6:hover a {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

from components.header import show_header # type: ignore
show_header()

st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)

# --- Database Connection ---
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="demo",
        user="postgres",
        password="Database@123"
    )
col1, col2, col3 = st.columns([9, 8, 9]) 

with col2:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                # Fetch the user by username
                cur.execute("SELECT user_id, username, password_hash, role FROM users WHERE username = %s", (username,))
                user = cur.fetchone()
                if user:
                    user_id, db_username, db_password_hash, role = user
                    if bcrypt.checkpw(password.encode(), db_password_hash.encode()):
                        # Update last_login (use UTC)
                        try:
                            cur.execute(
                                "UPDATE users SET last_login = %s WHERE user_id = %s",
                                (datetime.datetime.utcnow(), user_id)
                            )
                            conn.commit()
                        except Exception as e:
                            st.warning(f"Could not update last_login: {e}")
                        st.success("Login successful!")
                        st.session_state['user'] = db_username
                        st.session_state['role'] = role
                        # st.experimental_rerun()  # Uncomment if you want to reload or redirect
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Invalid username or password.")
                cur.close()
                conn.close()
            except Exception as e:
                st.error(f"Database error: {e}")

