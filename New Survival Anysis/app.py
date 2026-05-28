# app.py  ← the only file you need to run
#   conda activate osslab
#   cd "New Survival Anysis"
#   streamlit run app.py
import streamlit as st

pg = st.navigation([
    st.Page("DemoAppV2.3.py", title="Pipeline"),
    st.Page("Dashboard.py",   title="Dashboard"),
])
pg.run()