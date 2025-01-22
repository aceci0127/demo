from backend import AthenaSearch
import streamlit as st

# ----------- Streamlit UI & main logic -----------


st.title("ATHENA | Packaging DEMO")

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
                #PIPELINE FROM BACKEND
                athena_instance = AthenaSearch(prompt, index_body, index_abstract)
                answer = athena_instance.run_pipeline()

            # 3) Display the assistant's answer
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Append assistant response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": answer})
else:
    st.warning("Please enter a question before submitting.")