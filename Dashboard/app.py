# app.py  ← the only file you run
import streamlit as st

pg = st.navigation([
    st.Page("DemoAppV2.2.py", title="Pipeline"),
    st.Page("departure_sim.py",   title="Dashboard"),
    st.Page("DemoApp_Survival_Analysis.py", title="(Broken) Survival Analysis Pipeline"),
])
pg.run()