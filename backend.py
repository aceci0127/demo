from dotenv import load_dotenv
from llama_cloud.client import LlamaCloud
from pinecone import Pinecone
import openai
import os
from rdflib import Graph
from rdflib.namespace import Namespace
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph
import numpy as np
import json

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
client_deepseek = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
client_llama = LlamaCloud(token=LLAMA_CLOUD_API_KEY)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# set the configuration to connect to your Aura DB
AURA_DB_URI="neo4j+s://2886f391.databases.neo4j.io"
AURA_DB_USERNAME="neo4j"
AURA_DB_PWD="lrxu93e8exF48K_HTNtAi_FGrB9MOyC7J21uoNRxMaA"

auth_data = {'uri': AURA_DB_URI,
             'database': "neo4j",
             'user': AURA_DB_USERNAME,
             'pwd': AURA_DB_PWD}

# Define your custom mappings & store config
config = Neo4jStoreConfig(auth_data=auth_data,
                          #custom_prefixes=prefixes,
                          handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
                          batching=True)

from neo4j import GraphDatabase

uri = AURA_DB_URI
driver = GraphDatabase.driver(uri, auth=("neo4j", AURA_DB_PWD))

index_name = "papers-abstracts"
index = pc.Index(index_name)
index.describe_index_stats()

## PIPELINE

USER_QUERY = input("Ask Athena: ")

with open("prompts/cypher/LLMentityExtractor.txt", "r") as file:
    PROMPT_LLMentity_Extractor = file.read()

with open("prompts/cypher/CypherQueryBuilder.txt", "r") as file:
    PROMPT_CYPHER_QUERY_BUILDER = file.read()

with open("prompts/ANSWER.txt", "r") as file:
    PROMPT_answer= file.read()

def perform_response(query, vec_docs, graph_docs, prompt):
    # Generate a final response using the retrieved texts and user query
    
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\n------VECTOR RESULTS:{vec_docs}.\n\n------GRAPG RESULTS:{graph_docs}."}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content

    return answer

def generate_cypher(query, prompt, entities):
    
    response = client_deepseek.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"\n\nUSER QUERY{query}\n\nENTITIES EXTRACTED:{entities}"}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content

    return answer

def perform_embedding(text, model="text-embedding-3-large"):
        # Ensure the input text is a string
        if not isinstance(text, str):
            text = str(text)
        try:
            # Create an embedding vector using OpenAI's embedding model
            response = client_openai.embeddings.create(input=text, model=model)
            vector = response.data[0].embedding
            return vector
        except Exception as e:
            return None

def generate_entities(query, prompt):
    
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
    vec = perform_embedding(input_text)  # Get the embedding vector for the input text
    query_results = index.query(
        vector=vec,  # Use the embedding vector for the search
        top_k=6,  # Return the top 2 matches
        include_values=False,
        #filter={"id": {'$eq': "https://www.sciencedirect.com/science/article/pii/S2214157X24015958"}},
        include_metadata=True)
    # Extract metadata text and scores from the query results
    metadata_list = [match['metadata']['text'] for match in query_results['matches']]
    metadata_full = [match['metadata'] for match in query_results['matches']]
    scores = [match['score'] for match in query_results['matches']]
    return metadata_full

def generate_final_cypher(USER_QUERY, PROMPT_LLMentity_Extractor): 

    entities = generate_entities(USER_QUERY, PROMPT_LLMentity_Extractor)

    list_entities_llm = [item.strip() for item in entities.strip("[]").split(",")]


    CYPHER_MATERIAL = """ 
    MATCH (material:Material)
    RETURN material.embedding AS embedding, material.name AS name
    """

    CYPHER_TOPIC= """ 
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

    execute_query_mat(CYPHER_MATERIAL)
    execute_query_top(CYPHER_TOPIC)
    execute_query_app(CYPHER_APPLICATION)

    embeddings_dict = {
        "name": list_names,           # List of names
        "embedding": list_embeddings, # List of embeddings
        "entity type": list_entities  # List of entity types
    }


    EntitiesExtracted = ""

    for entity in list_entities_llm:
        embedded_query = perform_embedding(entity, model="text-embedding-3-small")

        # Parse embeddings if they are stored as strings
        def parse_embeddings(embeddings):
            return [json.loads(emb) if isinstance(emb, str) else emb for emb in embeddings]

        # Parse the embeddings in the dictionary
        parsed_embeddings = parse_embeddings(embeddings_dict["embedding"])
        names = embeddings_dict["name"]

        # Ensure query vector is parsed if it's a string
        query_vec = json.loads(embedded_query) if isinstance(embedded_query, str) else embedded_query

        # Cosine similarity function
        def cosine_similarity(vec1, vec2):
            v1, v2 = np.array(vec1), np.array(vec2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Calculate similarities and associate with names
        similarities = [
            {"name": name, "similarity": cosine_similarity(emb, query_vec)}
            for name, emb in zip(names, parsed_embeddings)
        ]

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Get top 3 most similar
        res = similarities[:1]

        EntitiesExtracted = EntitiesExtracted + "Entity Name: " + res[0]['name'] + "  \nEntity type is: " + embeddings_dict['entity type'][names.index(res[0]['name'])] + " \n"


    CYPHER = generate_cypher(USER_QUERY, PROMPT_CYPHER_QUERY_BUILDER, EntitiesExtracted)

    return CYPHER

def execute_final_cypher_query(query):
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            return record

     
final_query = generate_final_cypher(USER_QUERY, PROMPT_LLMentity_Extractor)

graphd_results = execute_final_cypher_query(final_query)

vectord_results = perform_search(USER_QUERY, index)

response = perform_response(USER_QUERY, vectord_results, graphd_results, PROMPT_answer)

print(response)