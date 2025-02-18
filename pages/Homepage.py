import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="ATHENA Demo",
    page_icon=":rocket:",
    layout="wide",
)

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Page background and container styling */
    body {
        background-color: #f0f2f6;
    }
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Header styles */
    .header-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #333333;
        margin-bottom: 0;
    }
    .section-header {
        font-size: 1.75rem;
        color: #0056b3;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.25rem;
    }
    .description {
        font-size: 1.1rem;
        color: #555555;
        line-height: 1.6;
    }
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section: Logo and Title in a row
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("images/Logo.png", width=100)
with col_title:
    st.markdown('<h1 class="header-title">ATHENA - A Demo Showcase</h1>', unsafe_allow_html=True)

# Introduction / Explanation
st.markdown(
    """
    <p class="description">
    Athena is our cutting-edge, AI-Search Assistant that showcases how technology can be tailored for specific industry challenges.
    Whether your focus is on Packaging, Med, or Thermal Barrier applications, Athena demonstrates how intelligent solutions can optimize processes, enhance quality, and drive efficiency.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# Packaging Section
st.markdown('<h2 class="section-header">Packaging</h2>', unsafe_allow_html=True)
st.markdown(
    """
    **Use Cases:** Quality control, supply chain optimization, and waste reduction.

    Athena leverages computer vision and predictive analytics to monitor packaging quality, optimize material usage, and streamline supply chains. 
    By providing real-time insights, manufacturers can reduce errors, cut waste, and save valuable resources.
    """
)

st.markdown("<hr>", unsafe_allow_html=True)

# Med Section
st.markdown('<h2 class="section-header">Med</h2>', unsafe_allow_html=True)
st.markdown(
    """
    **Use Cases:** Medical device diagnostics, patient monitoring, and workflow optimization.

    In the medical domain, Athena supports the reliability and performance of healthcare devices. 
    It analyzes operational data to enhance diagnostics, improve patient monitoring, and streamline clinical workflows, leading to better outcomes and increased efficiency.
    """
)

st.markdown("<hr>", unsafe_allow_html=True)

# Thermal Barrier Section
st.markdown('<h2 class="section-header">Thermal Barrier</h2>', unsafe_allow_html=True)
st.markdown(
    """
    **Use Cases:** Material performance analysis, failure prediction, and energy efficiency improvement.

    For thermal barrier applications, Athena applies advanced simulations and data analytics to monitor material degradation and predict failures. 
    This proactive approach helps maintain energy efficiency and prevents costly system downtimes.
    """
)

st.markdown("<hr>", unsafe_allow_html=True)

# Closing statement
st.markdown(
    """
    <p class="description">
    Explore the various demos to see how Athena can be adapted to revolutionize your industry-specific challenges!
    </p>
    """,
    unsafe_allow_html=True
)