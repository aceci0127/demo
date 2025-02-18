import streamlit as st

Home = st.Page("app.py", title="Homepage", icon=":material/home:")
Pack = st.Page("pages/Packaging.py", title="Packaging Demo")
Med = st.Page("pages/Medical.py", title="Medical Demo")

pg = st.navigation([Home, Pack, Med])

pg.run()