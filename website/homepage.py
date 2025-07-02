import streamlit as st
st.set_page_config(layout="wide")
from components.styles import apply_global_styles
apply_global_styles()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
from components.header import show_header 
import os
from docx import Document
show_header()


st.markdown("""
<style>
    MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 0rem; /* Adjust this value as needed */
    }
    
    h1:hover a, h2:hover a, h3:hover a, h4:hover a, h5:hover a, h6:hover a {
    display: none !important;
    }
    
</style>
""", unsafe_allow_html=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

st.markdown("<div style='text-align: center; font-size: 2.5em; font-weight: bold;'>ARGENTINA</div>", unsafe_allow_html=True)
st.markdown(
"""
<div style="background-color:#e8e1e1; color:#99111a; padding:15px; border-radius:8px; font-size:16px;">
    Please register or login to access all the indicators on the AI dashboard
</div>
""",
unsafe_allow_html=True
)

tabs = st.tabs([
    "Country Profile",
    "Climate Indicators",
    "Socio-economic Indicators",
    "Vulnerability Indicators",
    "Resilience Indicators",
    "Humanitarian Indicators"
])


with tabs[0]:
    left_col , right_col = st.columns([4,5])
    with left_col:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Mapa_Argentina_Tipos_clima_IGN.jpg/500px-Mapa_Argentina_Tipos_clima_IGN.jpg",
        caption="Mapa de los tipos de clima en Argentina", use_container_width=160 )
    with right_col:
        from pathlib import Path
        def load_docx(docx_file):
            doc = Document(docx_file)
            content = []
            for para in doc.paragraphs:
                text = para.text.strip()
                style = para.style.name
                if not text:
                    continue
                if style in ["Heading 1", "Heading 2", "Heading 3", "Heading 4"]:
                    content.append((style, text))
                else:
                    content.append(("bullet", text))
            return content

        def display_content(content):
            st.markdown("""
                <style>
                .heading1 { color: #1f4e79; font-size: 18px; font-weight: bold; margin-top: 30px;}
                .heading2 { color: #2e6da4; font-size: 16px; font-weight: bold; margin-top: 24px;}
                .heading3 { color: #3c8dbc; font-size: 14px; font-weight: 600; margin-top: 18px;}
                .heading4 { color: #5faee3; font-size: 14px; font-weight: 600; margin-top: 12px;}
                .custom-bullet { color: #333; font-size: 14px; margin-left: 20px; line-height: 1.6;}
                </style>
            """, unsafe_allow_html=True)
            for style, text in content:
                if style == "Heading 1":
                    st.markdown(f'<div class="heading1">{text}</div>', unsafe_allow_html=True)
                elif style == "Heading 2":
                    st.markdown(f'<div class="heading2">{text}</div>', unsafe_allow_html=True)
                elif style == "Heading 3":
                    st.markdown(f'<div class="heading3">{text}</div>', unsafe_allow_html=True)
                elif style == "Heading 4":
                    st.markdown(f'<div class="heading4">{text}</div>', unsafe_allow_html=True)
                elif style == "bullet":
                    st.markdown(f'<div class="custom-bullet">• {text}</div>', unsafe_allow_html=True)

        docx_path = Path("docs/country_prof.docx")
        doc_content = load_docx(docx_path)
        display_content(doc_content)
    
for i in range(1, 6):
    with tabs[i]:
        st.markdown("""
        <div style="
            pointer-events: none;
            opacity: 1;
            background: #e8e1e1;
            color: #99111a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;">
            <h3>Login to view the content</h3>
        </div>
        """, unsafe_allow_html=True)
        
