from backendv2 import AthenaSearch
from llm import LLM
import streamlit as st
import time

index_body = "papers-body-packaging"
index_abstract = "papers-abstracts"

# ----------- Streamlit UI & main logic -----------
st.title("C O P P E R")
st.info("The dataset is composed of only 100 scientific papers on Copper - ENGLISH ONLY🇬🇧")

# Initialize the conversation if not present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display existing conversation
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# On button click, run the pipeline
if prompt := st.chat_input("Ask Athena:"):
    # 1) Show the user's query as a chat bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message in session state
    st.session_state.conversation.append({"role": "user", "content": prompt})

    # 2) Run your pipeline
    with st.spinner("I'm thinking..."):
        start_time = time.time()  # Record the start time
        
        # PIPELINE FROM BACKEND (Athena)
        athena_instance = AthenaSearch(prompt, index_body, index_abstract, st.session_state.conversation)
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

    # Optionally, append the assistant responses to the conversation
    st.session_state.conversation.append({"role": "ai", "content": f"**ChatGPT:** {chatGPT_answer}\n\n**Athena:** {athena_answer}"})
else:
    st.warning("Please enter a question before submitting.")