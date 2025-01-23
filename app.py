from backend import AthenaSearch
import streamlit as st
import time

index_body = "papers-body-packaging"
index_abstract = "papers-abstracts"

# ----------- Streamlit UI & main logic -----------
st.image("images/Logo.png", width=78)
st.title("A T H E N A | Packaging DEMO")
st.info("Welcome to *ATHENA*, the AI-powered search engine for scientific papers on Packaging. The dataset is composed of only 100 scientific papers.  - ENGLISH ONLY🇬🇧")

# Initialize the conversation if not present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

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
    with st.spinner("Thinking..."):
        start_time = time.time()  # Record the start time
        
        # PIPELINE FROM BACKEND
        athena_instance = AthenaSearch(prompt, index_body, index_abstract, st.session_state.conversation)
        answer, paper_id = athena_instance.run_pipeline()
  # Assume the pipeline returns answer and paper_id

        end_time = time.time()  # Record the end time
        response_time = end_time - start_time  # Calculate response time

    # 3) Display the assistant's answer
    with st.chat_message("ai"):
        st.markdown(answer)

    # Append assistant response to conversation
    st.session_state.conversation.append({"role": "ai", "content": answer})

else:
    st.warning("Please enter a question before submitting.")