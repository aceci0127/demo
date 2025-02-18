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
    /* Global background */
    body {
        background-color: #f0f2f6;
    }
    /* Container styling */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Header styling */
    .header-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #333333;
        margin-bottom: 0.25rem;
    }
    .description {
        font-size: 1.1rem;
        color: #555555;
        line-height: 1.6;
    }
    /* Card styling for use case sections */
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
    }
    .card h3 {
        color: #0056b3;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .card ul {
        list-style-type: disc;
        padding-left: 1.5rem;
        color: #555;
    }
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
    /* Remove default link styles for cards */
    a {
        text-decoration: none;
        color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section: Logo and Title side by side
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("images/Logo.png", width=100)
with col_title:
    st.markdown('<h1 class="header-title">ATHENA - A Demo Showcase</h1>', unsafe_allow_html=True)

# Brief description
st.markdown(
    """
    <p class="description">
    Athena is our cutting-edge, AI-Search Assistant tailored for specific industry challenges. Choose a demo below to explore its capabilities.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# Display use cases as clickable cards in a three-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <a href="/Packaging">
          <div class="card">
              <h3>Packaging</h3>
              <ul>
                  <li>Quality Control</li>
                  <li>Supply Chain Optimization</li>
                  <li>Waste Reduction</li>
              </ul>
          </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <a href="/Med">
          <div class="card">
              <h3>Med</h3>
              <ul>
                  <li>Medical Device Diagnostics</li>
                  <li>Patient Monitoring</li>
                  <li>Workflow Optimization</li>
              </ul>
          </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <a href="/ThermalBarrier">
          <div class="card">
              <h3>Thermal Barrier</h3>
              <ul>
                  <li>Material Performance Analysis</li>
                  <li>Failure Prediction</li>
                  <li>Energy Efficiency Improvement</li>
              </ul>
          </div>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# Closing statement
st.markdown(
    """
    <p class="description">
    Explore the demos to see how Athena can revolutionize your industry-specific challenges!
    </p>
    """,
    unsafe_allow_html=True
)