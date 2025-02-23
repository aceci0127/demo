import streamlit as st

Home = st.Page("pages/Homepage.py", title="Homepage", icon=":material/home:")
Pack = st.Page("pages/Packaging.py", title="Packaging")
Cop = st.Page("pages/Copper2.py", title="Copper")
Ther = st.Page("pages/ThermalB.py", title="Thermal Barrier")


pg = st.navigation([Home, Pack, Cop, Ther])


pg.run()