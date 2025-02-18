import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="Your Startup - Athena Demo",
    page_icon=":rocket:",
    layout="wide",
)

# Display the startup logo and name
logo_path = "images/Logo.png"  # Replace with your logo's correct directory
st.image(logo_path, width=100)
st.title("Your Startup Name")

# Explanation of the Athena demo version
st.markdown("""
## Welcome to the Athena Demo

Athena is our cutting-edge, AI-powered demo that showcases how technology can be tailored for specific industry challenges. Whether your focus is on Packaging, Med, or Thermal Barrier applications, Athena demonstrates how intelligent solutions can optimize processes, enhance quality, and drive efficiency.
""")

# Sidebar for domain selection
st.sidebar.title("Select a Domain")
domain = st.sidebar.radio("Choose a demo domain:", 
                          ["Packaging", "Med", "Thermal Barrier"])

# Display domain-specific Athena demos
if domain == "Packaging":
    st.header("Athena in Packaging")
    st.write("""
    **Use Cases:** Quality control, supply chain optimization, and waste reduction.
    
    Athena leverages computer vision and predictive analytics to monitor packaging quality, optimize material usage, and streamline supply chains. By providing real-time insights, manufacturers can reduce errors, cut waste, and save valuable resources.
    """)
elif domain == "Med":
    st.header("Athena in Med")
    st.write("""
    **Use Cases:** Medical device diagnostics, patient monitoring, and workflow optimization.
    
    In the medical domain, Athena supports the reliability and performance of healthcare devices. It analyzes operational data to enhance diagnostics, improve patient monitoring, and streamline clinical workflows, leading to better outcomes and increased efficiency.
    """)
elif domain == "Thermal Barrier":
    st.header("Athena in Thermal Barrier")
    st.write("""
    **Use Cases:** Material performance analysis, failure prediction, and energy efficiency improvement.
    
    For thermal barrier applications, Athena applies advanced simulations and data analytics to monitor material degradation and predict failures. This proactive approach helps maintain energy efficiency and prevents costly system downtimes.
    """)

st.markdown("---")
st.markdown("Explore the various demos to see how Athena can be adapted to revolutionize your industry-specific challenges!")