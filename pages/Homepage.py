import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="ATHENA Demo",
    page_icon=":rocket:",
    layout="wide",
)

# Inject custom CSS for styling
st.markdown("""
    <style>
        /* Global styling */
        body {
            background-color: #f0f2f6;
            font-family: 'Roboto', sans-serif;
        }
        /* Main container styling */
        .main .block-container {
            background-color: #ffffff;
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        /* Header styling */
        .header-title {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 800;
            color: #333;
            margin-bottom: 0.5rem;
        }
        .description {
            font-size: 1.15rem;
            color: #555;
            line-height: 1.6;
        }
        /* Card styling */
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }
        .card h3 {
            color: #0073e6;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }
        .card ul {
            list-style-type: disc;
            padding-left: 1.5rem;
            color: #666;
            margin: 0;
        }
        /* Remove underline from all elements inside links */
        a, a:link, a:visited, a:hover, a:active, a * {
            text-decoration: none !important;
            color: inherit;
        }
        hr {
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 2.5rem 0;
        }
        /* Responsive adjustments for smaller screens */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1.5rem;
            }
            .card {
                margin: 1rem 0;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header Section: Logo and Title side by side
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("images/Logo.png", width=100)
with col_title:
    st.markdown('<h1 class="header-title">A T H E N A - Demo Showcase \nAccelerate Innovation</h1>', unsafe_allow_html=True)

# Brief description
st.markdown("""
    <p class="description">
    Athena is our AI-Search Assistant that allows you to interact with thousand of scientic documents.
    </p>
    <p class="description">
    Use the navigation bar on the left or click on the cards below to explore the demos.
    </p>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Display use cases as clickable cards in a three-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <a href="/Packaging">
          <div class="card">
              <h3>Packaging</h3>
              <ul>
                <li>Packaging Techniques</li>
                <li>Packaging on Demand</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href="/Medical.py">
          <div class="card">
              <h3>Med</h3>
              <ul>
                <li>Theme 1</li>
                <li>Theme 2</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <a href="/ThermalB.py">
          <div class="card">
              <h3>Thermal Barrier</h3>
              <ul>
                <li>Theme 1</li>
                <li>Theme 2</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Closing statement
st.markdown("""
    <p class="description">
    Explore the demos to see how Athena can revolutionize your industry-specific challenges!
    </p>
""", unsafe_allow_html=True)