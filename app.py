import streamlit as st
from dotenv import load_dotenv
from llama_cloud.client import LlamaCloud
from pinecone import Pinecone
import openai
import os
from rdflib import Graph
from rdflib.namespace import Namespace
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
import numpy as np
import json
from neo4j import GraphDatabase 

# 1) Load environment variables from .env file
load_dotenv()

# 2) Setup keys and clients
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
client_deepseek = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)

# 3) Neo4j/Aura DB connection
AURA_DB_URI = "neo4j+s://2886f391.databases.neo4j.io"  # change to your correct Neo4j Aura URI
AURA_DB_USERNAME = "neo4j"
AURA_DB_PWD = "lrxu93e8exF48K_HTNtAi_FGrB9MOyC7J21uoNRxMaA"

auth_data = {
    'uri': AURA_DB_URI,
    'database': "neo4j",
    'user': AURA_DB_USERNAME,
    'pwd': AURA_DB_PWD
}

config = Neo4jStoreConfig(
    auth_data=auth_data,
    handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
    batching=True
)

driver = GraphDatabase.driver(AURA_DB_URI, auth=("neo4j", AURA_DB_PWD))

# 4) Pinecone index
index_name = "papers-abstracts"
index = pc.Index(index_name)
# Optional: describe index stats
# index.describe_index_stats()

# 5) Load prompt templates
with open("prompts/cypher/LLMentityExtractor.txt", "r") as file:
    PROMPT_LLMentity_Extractor = file.read()

with open("prompts/cypher/CypherQueryBuilder.txt", "r") as file:
    PROMPT_CYPHER_QUERY_BUILDER = file.read()

with open("prompts/ANSWER.txt", "r") as file:
    PROMPT_answer = file.read()

# ----------- Helper functions -----------

def perform_response(query, vec_docs, graph_docs, prompt):
    """
    Generate a final response using the retrieved texts and user query
    """
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": (
                    f"\n\n\n-----QUERY:{query}"
                    f"\n\n------VECTOR RESULTS:{vec_docs}."
                    f"\n\n------GRAPG RESULTS:{graph_docs}."
                )
            },
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer

def generate_cypher(query, prompt, entities):
    """
    Use the DeepSeek coder model to build Cypher queries
    based on user query and extracted entities.
    """
    response = client_deepseek.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": f"\n\nUSER QUERY{query}\n\nENTITIES EXTRACTED:{entities}"
            }
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer

def perform_embedding(text, model="text-embedding-3-large"):
    """
    Create an embedding vector using OpenAI's embedding model
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        response = client_openai.embeddings.create(input=text, model=model)
        vector = response.data[0].embedding
        return vector
    except Exception:
        return None

def generate_entities(query, prompt):
    """
    Extract entities from user query using the DeepSeek coder model
    """
    response = client_deepseek.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"\n\nUSER QUERY{query}"}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer

def perform_search(input_text, index):
    """
    Performs a similarity search on Pinecone index using the input_text
    """
    vec = perform_embedding(input_text)  # get the embedding vector
    query_results = index.query(
        vector=vec,
        top_k=6,
        include_values=False,
        include_metadata=True
    )
    # Extract metadata text and scores from the query results
    metadata_list = [match['metadata']['text'] for match in query_results['matches']]
    metadata_full = [match['metadata'] for match in query_results['matches']]
    scores = [match['score'] for match in query_results['matches']]
    return metadata_full

def generate_final_cypher(user_query, entity_extractor_prompt):
    """
    1) Extract entity candidates from user query
    2) Retrieve relevant nodes from Neo4j
    3) Compute similarity to find top matching nodes
    4) Build a final Cypher query
    """
    entities = generate_entities(user_query, entity_extractor_prompt)
    list_entities_llm = [item.strip() for item in entities.strip("[]").split(",")]

    # Cypher queries to retrieve stored embeddings from your Neo4j
    CYPHER_MATERIAL = """ 
    MATCH (material:Material)
    RETURN material.embedding AS embedding, material.name AS name
    """
    CYPHER_TOPIC = """ 
    MATCH (topic:Topic)
    RETURN topic.embedding AS embedding, topic.name AS name
    """
    CYPHER_APPLICATION = """ 
    MATCH (application:Application)
    RETURN application.embedding AS embedding, application.name AS name
    """

    list_embeddings = []
    list_names = []
    list_entities = []

    def execute_query_mat(query):
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Material')

    def execute_query_top(query):
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Topic')

    def execute_query_app(query):
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Application')

    # Execute queries to fill in all possible embeddings
    execute_query_mat(CYPHER_MATERIAL)
    execute_query_top(CYPHER_TOPIC)
    execute_query_app(CYPHER_APPLICATION)

    embeddings_dict = {
        "name": list_names,
        "embedding": list_embeddings,
        "entity type": list_entities
    }

    EntitiesExtracted = ""

    for entity in list_entities_llm:
        embedded_query = perform_embedding(entity, model="text-embedding-3-small")

        # parse embeddings if they are stored as strings
        def parse_embeddings(embeddings):
            return [
                json.loads(emb) if isinstance(emb, str) else emb
                for emb in embeddings
            ]

        parsed_embeddings = parse_embeddings(embeddings_dict["embedding"])
        names = embeddings_dict["name"]

        query_vec = (
            json.loads(embedded_query) if isinstance(embedded_query, str)
            else embedded_query
        )

        # Cosine similarity
        def cosine_similarity(vec1, vec2):
            v1, v2 = np.array(vec1), np.array(vec2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        similarities = []
        for name, emb in zip(names, parsed_embeddings):
            sim = cosine_similarity(emb, query_vec)
            similarities.append({"name": name, "similarity": sim})

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        # top match
        top_match = similarities[0]
        top_name = top_match['name']
        top_index = names.index(top_name)
        entity_type = embeddings_dict['entity type'][top_index]

        EntitiesExtracted += (
            f"Entity Name: {top_name}\nEntity type is: {entity_type}\n"
        )

    # Now call the function to build an actual Cypher query from these extracted entities
    CYPHER = generate_cypher(user_query, PROMPT_CYPHER_QUERY_BUILDER, EntitiesExtracted)
    return CYPHER

def execute_final_cypher_query(query):
    """
    Execute the final Cypher query in Neo4j
    """
    with driver.session() as session:
        result = session.run(query)
        # For demonstration, let's return everything we get from the first record
        records = result.data()
        return records

# ----------- Streamlit UI & main logic -----------


st.title("ATHENA | Chatbot")

# Initialize the conversation if not present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display existing conversation
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User inputs a question via text_input
user_query = st.text_input("Ask Athena:")

    # On button click, run the pipeline
if prompt := st.chat_input("Ask frenchbot anything..."):
            # 1) Show the user's query as a chat bubble
            with st.chat_message("user"):
                st.markdown(user_query)

            # Save user message in session state
            st.session_state.conversation.append({"role": "user", "content": user_query})

            # 2) Run your pipeline
            with st.spinner("Thinking..."):
                final_query = generate_final_cypher(user_query, PROMPT_LLMentity_Extractor)
                graphd_results = execute_final_cypher_query(final_query)
                vectord_results = perform_search(user_query, index)
                response = perform_response(user_query, vectord_results, graphd_results, PROMPT_answer)

            # 3) Display the assistant's answer
            with st.chat_message("assistant"):
                st.markdown(response)

            # Append assistant response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": response})
else:
    st.warning("Please enter a question before submitting.")