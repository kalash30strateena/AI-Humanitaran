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
        st.write(" ")
        docx_path = "docs/country_profile_data.docx"  # Your file path
        def docx_to_markdown(doc):
            md_lines = []
            for para in doc.paragraphs:
                md_para = ""
                for run in para.runs:
                    text = run.text.replace('\n', ' ')
                    if not text:
                        continue
                    # Apply formatting
                    if run.bold:
                        text = f"**{text}**"
                    if run.italic:
                        text = f"*{text}*"
                    if run.underline:
                        text = f"__{text}__"
                    md_para += text
                md_lines.append(md_para)
            return "\n".join(md_lines)

        if os.path.exists(docx_path):
            doc = Document(docx_path)
            markdown_content = docx_to_markdown(doc)
            st.markdown(markdown_content)
        else:
            st.error(f"File not found: {docx_path}")
    
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
        
