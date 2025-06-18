import streamlit as st
def logged_header():
    if "language" not in st.session_state:
        st.session_state["language"] = "English"

    st.markdown("""
    <style>
    /* Header styling */
    .custom-header {{
        background-color: #ffffff;
        font-size: 20.5px;
        padding: 0px 0px;
        border-bottom: 2px solid #eaeaea;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0px;
    }}
    .custom-header img {{
        height: 100px;
    }}
    .custom-header .header-center {{
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .custom-header .header-links {{
        display: flex;
        align-items: center;
    }}
    .custom-header .header-links a {{
        color: #7a2e2b;
        margin-left: 10px;
        font-size: 16px;
        text-decoration: none;
        padding: 8px 8px;
        border-radius: 18px;
        transition: background 0.2s, color 0.2s;
    }}
    .custom-header .header-links a:hover {{
        background-color: #b22222;
        color: #fff !important;
        border-radius: 18px;
        text-decoration: none;
    }}
    .language-select {{
        margin-right: 30px;
        font-size: 16px;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #ccc;
        color: #7a2e2b;
        background-color: #f9f9f9;
    }}
    .stTabs [data-baseweb="tab"] {{
        margin-right: 7px !important;
    }}
    .stTabs [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {{
        font-size: 20px !important;
        color: #7a2e2b;
        transition: color 0.2s;
    }}
    /* Active (clicked) tab styling */
    .stTabs [data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {{
        color: #e53935 !important; 
    }}
    /* Weather forecast card styling */
    .forecast-card {{
        background: #62a1c7;
        color: #fff;
        border-radius: 10px;
        padding: 6px 6px 6px 6px;
        margin: 2px;
        text-align: center;
        min-width: 200px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        font-family: 'Segoe UI', sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .forecast-card .day {{
        background: #3a4d5c;
        border-radius: 2px;
        display: inline-block;
        font-size: 11px;
        margin-bottom: 2px;
        font-weight: bold;
    }}
    .forecast-card .date {{
        font-size: 12px;
        color: #050505;
        margin-bottom: 10px;
    }}
    .forecast-card .icon {{
        font-size: 40px;
    }}
    .forecast-card .desc {{
        font-size: 12px;
        margin-bottom: 2px;
        color: #e0e0e0;
    }}
    .forecast-card .temp {{
        font-size: 15px;
        margin-top: 2px;
        font-weight: bold;
    }}
    .forecast-summary-header {{
        font-size: 17.5px !important;
        color: #0a0a0a; 
    }}
    </style>

    <div class="custom-header">
        <div>
    <a href="/" target="_self">
        <img src="https://cruzroja.org.ar/observatorio-humanitario/wp-content/uploads/2020/09/logos-cra-mr-2023.png" alt="Logo" width="400">
    </a>
    </div>
        <div class="header-center"></div>
        <div class="header-links">
            <form action="" method="get" id="lang-form">
                <select name="language" class="language-select" onchange="document.getElementById('lang-form').submit();">
                    <option value="English" {eng_selected}>English</option>
                    <option value="Español" {esp_selected}>Español</option>
                </select>
            </form>
            <a href="/" target="_self">Logout</a>
        </div>
    </div>
    """.format(
        eng_selected="selected" if st.session_state.get("language", "English") == "English" else "",
        esp_selected="selected" if st.session_state.get("language", "English") == "Español" else "",
    ), unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)