import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="ATHENA Demo",
    page_icon=":rocket:",
    layout="wide",
)

Home = st.Page("app.py", title="Homepage", icon=":material/add_circle:")
Pack = st.Page("pages/Packaging.py", title="Packaging Demo", icon=":material/delete:")

st.navigation([Home, Pack])

# Display the startup logo and name
logo_path = "images/Logo.png"  # Replace with your logo's correct directory
st.image(logo_path, width=100)
st.title("ATHENA")

# Explanation of the Athena demo version
st.markdown("""
## Welcome to the Athena Demo

Athena is our cutting-edge, AI-powered demo that showcases how technology can be tailored for specific industry challenges. Whether your focus is on Packaging, Med, or Thermal Barrier applications, Athena demonstrates how intelligent solutions can optimize processes, enhance quality, and drive efficiency.
""")

# Sidebar navigation with anchors for each domain section
st.sidebar.title("Jump to a Domain")
st.sidebar.markdown("[Packaging](#packaging)")
st.sidebar.markdown("[Med](#med)")
st.sidebar.markdown("[Thermal-Barrier](#thermal-barrier)")

# Packaging section
st.markdown("---")
st.header("Packaging")
st.markdown("""
**Use Cases:** Quality control, supply chain optimization, and waste reduction.

Athena leverages computer vision and predictive analytics to monitor packaging quality, optimize material usage, and streamline supply chains. By providing real-time insights, manufacturers can reduce errors, cut waste, and save valuable resources.
""")

# Med section
st.markdown("---")
st.header("Med")
st.markdown("""
**Use Cases:** Medical device diagnostics, patient monitoring, and workflow optimization.

In the medical domain, Athena supports the reliability and performance of healthcare devices. It analyzes operational data to enhance diagnostics, improve patient monitoring, and streamline clinical workflows, leading to better outcomes and increased efficiency.
""")

# Thermal Barrier section
st.markdown("---")
st.header("Thermal Barrier")
st.markdown("""
**Use Cases:** Material performance analysis, failure prediction, and energy efficiency improvement.

For thermal barrier applications, Athena applies advanced simulations and data analytics to monitor material degradation and predict failures. This proactive approach helps maintain energy efficiency and prevents costly system downtimes.
""")

st.markdown("---")
st.markdown("Explore the various demos to see how Athena can be adapted to revolutionize your industry-specific challenges!")