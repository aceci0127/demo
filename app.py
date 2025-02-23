import streamlit as st

Home = st.Page("pages/Homepage.py", title="Homepage", icon=":material/home:")
Pack = st.Page("pages/Packaging.py", title="Packaging")
Med = st.Page("pages/Medical.py", title="Medical")
Ther = st.Page("pages/ThermalB.py", title="Thermal Barrier")

pg = st.navigation([Home, Pack, Med, Ther])

pg.run()