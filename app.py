from backend import AthenaSearch
import streamlit as st

index_body = "papers-body-packaging"
index_abstract = "papers-abstracts"

# ----------- Streamlit UI & main logic -----------


st.title("ATHENA | Packaging DEMO")

# Initialize the conversation if not present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

import csv
import time
import os

# File path for analytics
analytics_file = "analytics.csv"

# Function to initialize the analytics file if not present
def initialize_analytics_file():
    if not os.path.exists(analytics_file):
        with open(analytics_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["paper_id", "question", "answer", "response_time (s)"])

# Call the initializer to ensure file exists
initialize_analytics_file()

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
        athena_instance = AthenaSearch(prompt, index_body, index_abstract)
        answer, paper_id = athena_instance.run_pipeline()
  # Assume the pipeline returns answer and paper_id

        end_time = time.time()  # Record the end time
        response_time = end_time - start_time  # Calculate response time

    # 3) Display the assistant's answer
    with st.chat_message("ai"):
        st.markdown(answer)

    # Append assistant response to conversation
    st.session_state.conversation.append({"role": "ai", "content": answer})

    # Save analytics data
    analytics_data = {
        "paper_id": paper_id,
        "question": prompt,
        "answer": answer,
        "response_time (s)": round(response_time, 2),
    }
    with open(analytics_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["paper_id", "question", "answer", "response_time (s)"])
        writer.writerow(analytics_data)

else:
    st.warning("Please enter a question before submitting.")