import streamlit as st

Home = st.Page("pages/Homepage.py", title="Homepage", icon=":material/home:")
Pack = st.Page("pages/Packaging.py", title="Packaging")
Med = st.Page("pages/Medical.py", title="Medical")
Ther = st.Page("pages/ThermalB.py", title="Thermal Barrier")
st.markdown('<link href="https://fonts.cdnfonts.com/css/glacial-indifference" rel="stylesheet">', unsafe_allow_html=True)

pg = st.navigation([Home, Pack, Med, Ther])

pg.run()