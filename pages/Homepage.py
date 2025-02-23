import streamlit as st

# Configura la pagina
st.set_page_config(
    page_title="ATHENA Demo",
    page_icon=":rocket:",
    layout="wide",
)

# Includi il font personalizzato (Glacial Indifference) da un CDN
st.markdown('<link href="https://fonts.cdnfonts.com/css/glacial-indifference" rel="stylesheet">', unsafe_allow_html=True)

# Applica CSS personalizzato per lo stile
st.markdown("""
    <style>
        /* Stile globale con un leggero gradiente di sfondo */
        body {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            font-family: 'Glacial Indifference', sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }
        /* Stile del contenitore principale */
        .main .block-container {
            background-color: #ffffff;
            padding: 3rem 4rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        /* Stile dell'intestazione */
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
        /* Stile delle schede */
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
        /* Rimuovi la sottolineatura dai link e dai loro elementi */
        a, a:link, a:visited, a:hover, a:active, a * {
            text-decoration: none !important;
            color: inherit;
        }
        hr {
            border: none;
            border-top: 1px solid #dce2e8;
            margin: 3rem 0;
        }
        /* Adattamenti per dispositivi mobili */
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

# Sezione Intestazione: Logo e Titolo affiancati
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("images/Logo.png", width=100)
with col_title:
    st.markdown('<h1 class="header-title">A T H E N A - Demo</h1> <h2>Esplora i confimi della conoscenza attraverso conversazioni</h2>', unsafe_allow_html=True)

# Breve descrizione
st.markdown("""
    <p class="description">
    Athena è il nostro assistente di ricerca AI che ti permette di interagire con migliaia di documenti scientifici.
    </p>
    <p class="description">
    Utilizza la barra di navigazione a sinistra o clicca sulle schede qui sotto per esplorare le demo.
    </p>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Visualizza i casi d'uso come schede cliccabili in un layout a tre colonne
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <a href="/Packaging">
          <div class="card">
              <h3>Packaging</h3>
              <ul>
                <li>Tecniche di Imballaggio</li>
                <li>Packaging on Demand</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href="/Medical.py">
          <div class="card">
              <h3>Replace...</h3>
              <ul>
                <li>Tema 1</li>
                <li>Tema 2</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <a href="/ThermalB.py">
          <div class="card">
              <h3>Barriera Termica</h3>
              <ul>
                <li>Tema 1</li>
                <li>Tema 2</li>
              </ul>
          </div>
        </a>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Messaggio di chiusura
st.markdown("""
    <p class="description">
    Esplora le demo per scoprire come Athena può rivoluzionare le sfide specifiche del tuo settore!
    </p>
""", unsafe_allow_html=True)