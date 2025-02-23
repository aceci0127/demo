import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="ATHENA Demo",
    page_icon=":rocket:",
    layout="wide",
)

# Include the custom font (Glacial Indifference) from a CDN
st.markdown('<link href="https://fonts.cdnfonts.com/css/glacial-indifference" rel="stylesheet">', unsafe_allow_html=True)

# Inject custom CSS for styling
st.markdown("""
    <style>
        /* Global styling with a soft gradient background */
        body {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            font-family: 'Glacial Indifference', sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }
        /* Main container styling */
        .main .block-container {
            background-color: #ffffff;
            padding: 3rem 4rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        /* Header styling */
        .header-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #222;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        h2 {
            font-size: 1.75rem;
            color: #555;
            margin-top: 0;
        }
        .description {
            font-size: 1.2rem;
            color: #444;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        /* Card styling */
        .card {
            background-color: #fafafa;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid #eaeaea;
        }
        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        }
        .card h3 {
            font-size: 1.5rem;
            color: #0073e6;
            margin-bottom: 1rem;
            padding-bottom: 0.25rem;
            border-bottom: 2px solid #dfe6f0;
        }
        .card ul {
            list-style-type: disc;
            padding-left: 1.5rem;
            color: #666;
            margin: 0;
        }
        /* Remove underline from links and their children */
        a, a:link, a:visited, a:hover, a:active, a * {
            text-decoration: none !important;
            color: inherit;
        }
        hr {
            border: none;
            border-top: 1px solid #dce2e8;
            margin: 3rem 0;
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 2rem;
            }
            .card {
                margin: 1rem 0;
            }
            .header-title {
                font-size: 2.5rem;
            }
            h2 {
                font-size: 1.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header Section: Logo and Title side by side
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("images/Logo.png", width=100)
with col_title:
    st.markdown('<h1 class="header-title">A T H E N A - Demo Showcase</h1> <h2>Surf the Cutting Edge of Knowledge through Conversations</h2>', unsafe_allow_html=True)

# Brief description
st.markdown("""
    <p class="description">
    Athena is our AI-Search Assistant that allows you to interact with thousands of scientific documents.
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
              <h3>Medical</h3>
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