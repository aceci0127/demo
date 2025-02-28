from backend import AthenaSearch
from backendv2 import AthenaSearchv2
from llm import LLM
import streamlit as st
import time

index_body = "packaging-demo"
index_abstract = "packaging-demo-abstracts"

# ----------- Streamlit UI & main logic -----------
st.title("P A C K A G I N G")
st.info("Il Database è composto da 1000 articoli scientifici relativi alle techniche di Packaging e Packaging on Demand.")

# Use a unique session state key for this page
if 'conversation_packaging' not in st.session_state:
    st.session_state.conversation_packaging = []

# Display existing conversation for the Packaging page
for msg in st.session_state.conversation_packaging:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# On button click, run the pipeline with a unique widget key
if prompt := st.chat_input("Ask Athena:", key="packaging_chat_input"):
    # 1) Show the user's query as a chat bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message in the page-specific session state
    st.session_state.conversation_packaging.append({"role": "user", "content": prompt})

    # 2) Run your pipeline
    with st.spinner("I'm thinking..."):
        start_time = time.time()  # Record the start time
        
        # PIPELINE FROM BACKEND (Athena)
        athena_instance = AthenaSearch(prompt, index_body, index_abstract, st.session_state.conversation_packaging)
        athena_answer = athena_instance.run_pipeline()

        if False:    
        # PIPELINE FROM BACKEND (Athena)
            athena_instancev2 = AthenaSearchv2(prompt, index_body, index_abstract, st.session_state.conversation_packaging)
            athena_answerv2 = athena_instancev2.run_pipeline()

        if False:       
        # PIPELINE FROM GPT (ChatGPT)
            chatGPT = LLM(prompt)
            chatGPT_answer = chatGPT.run_pipeline()

        end_time = time.time()  # Record the end time
        response_time = end_time - start_time  # Calculate response time

    if False:
    # 3) Display both answers side by side in two columns
        cols = st.columns(2)
        with cols[1]:
            st.subheader("ChatGPT")
            st.markdown(chatGPT_answer)
        with cols[0]:
            st.subheader("ATHENA")
            st.markdown(athena_answer)
    st.markdown(athena_answer)

    # Append the assistant responses to the page-specific conversation
    st.session_state.conversation_packaging.append({
        "role": "ai",
        "content": f"**Answer:** {athena_answer}"
    })
else:
    st.warning("Please enter a question before submitting.")