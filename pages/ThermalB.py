from backend import AthenaSearch
from llm import LLM
import streamlit as st
import time

index_body = "thermalbarrier-demo"
index_abstract = "thermalbarrier-demo-abstracts"

# ----------- Streamlit UI & main logic -----------
st.title("T H E R M A L  B A R R I E R")
st.info("Il database è composto da 100 articoli scientifici sulle Barriere Termiche")

# Use a unique session state key for this page
if 'conversation_thermal' not in st.session_state:
    st.session_state.conversation_thermal = []

# Display existing conversation for the Thermal Barrier page
for msg in st.session_state.conversation_thermal:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# On button click, run the pipeline with a unique widget key
if prompt := st.chat_input("Ask Athena:", key="thermal_chat_input"):
    # 1) Show the user's query as a chat bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message in the page-specific session state
    st.session_state.conversation_thermal.append({"role": "user", "content": prompt})

    # 2) Run your pipeline
    with st.spinner("I'm thinking..."):
        start_time = time.time()  # Record the start time
        
        # PIPELINE FROM BACKEND (Athena)
        athena_instance = AthenaSearch(prompt, index_body, index_abstract, st.session_state.conversation_thermal)
        athena_answer, paper_id = athena_instance.run_pipeline()

        # PIPELINE FROM GPT (ChatGPT)
        chatGPT = LLM(prompt)
        chatGPT_answer = chatGPT.run_pipeline()

        end_time = time.time()  # Record the end time
        response_time = end_time - start_time  # Calculate response time

    # 3) Display both answers side by side in two columns
    cols = st.columns(2)
    with cols[0]:
        st.subheader("ChatGPT Answer")
        st.markdown(chatGPT_answer)
    with cols[1]:
        st.subheader("Athena Answer")
        st.markdown(athena_answer)

    # Append the assistant responses to the page-specific conversation
    st.session_state.conversation_thermal.append({
        "role": "ai",
        "content": f"**ChatGPT:** {chatGPT_answer}\n\n**Athena:** {athena_answer}"
    })
else:
    st.warning("Please enter a question before submitting.")