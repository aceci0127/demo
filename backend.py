USER_QUERY = "Which are the possible applications of Lignocellulosic bioplastics?"

from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import os
import numpy as np
import json
from neo4j import GraphDatabase

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
client_deepseek = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# set the configuration to connect to your Aura DB
AURA_DB_URI="neo4j+s://2886f391.databases.neo4j.io"
AURA_DB_USERNAME="neo4j"
AURA_DB_PWD="lrxu93e8exF48K_HTNtAi_FGrB9MOyC7J21uoNRxMaA"

uri = AURA_DB_URI
driver = GraphDatabase.driver(uri, auth=("neo4j", AURA_DB_PWD))

index_name_body = "papers-body-packaging"
index_body = pc.Index(index_name_body)
index_body.describe_index_stats()

index_name_abstract = "papers-abstracts"
index_abstract = pc.Index(index_name_abstract)
index_abstract.describe_index_stats()

with open("prompts/cypher/LLMentityExtractor.txt", "r") as file:
    PROMPT_LLMentity_Extractor = file.read()

with open("prompts/cypher/CypherQueryBuilder.txt", "r") as file:
    PROMPT_CYPHER_QUERY_BUILDER = file.read()

with open("prompts/ANSWER.txt", "r") as file:
    PROMPT_answer= file.read()

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
        include_metadata=True)
    # Extract metadata text and scores from the query results
    metadata_list = [match['metadata']['text'] for match in query_results['matches']]
    metadata_full = [match['metadata'] for match in query_results['matches']]
    scores = [match['score'] for match in query_results['matches']]
    return metadata_full

def perform_search_id(input_text, index):
    vec = perform_embedding(input_text)  # Get the embedding vector for the input text
    query_results = index.query(
        vector=vec,  # Use the embedding vector for the search
        top_k=6,  # Return the top 2 matches
        include_values=False,
        include_metadata=True)
    # Extract metadata text and scores from the query results
    metadata_id = [match['metadata']['id'] for match in query_results['matches']]
    return metadata_id

def perform_response(query, results, prompt):
    # Generate a final response using the retrieved texts and user query
    
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\n------VECTOR RESULTS:{results}."}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer

ENTITIES_GENERATED = generate_entities(USER_QUERY, PROMPT_LLMentity_Extractor)
LIST_OF_ENTITY = [item.strip() for item in ENTITIES_GENERATED.strip("[]").split(",")]

def generate_embed_dictionary():
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


    with driver.session() as session:
        result = session.run(CYPHER_MATERIAL)
        for record in result:
            list_embeddings.append(record['embedding'])
            list_names.append(record['name'])
            list_entities.append('Material')

    with driver.session() as session:
        result = session.run(CYPHER_TOPIC)
        for record in result:
            list_embeddings.append(record['embedding'])
            list_names.append(record['name'])
            list_entities.append('Topic')


    with driver.session() as session:
        result = session.run(CYPHER_APPLICATION)
        for record in result:
            list_embeddings.append(record['embedding'])
            list_names.append(record['name'])
            list_entities.append('Application')

    # Example dictionary of names and embeddings
    embeddings_dict = {
        "name": list_names,           # List of names
        "embedding": list_embeddings, # List of embeddings
        "entity type": list_entities  # List of entity types
    }

    return embeddings_dict

DICT = generate_embed_dictionary()

def extract_entities(LIST, DICT):
    EntitiesExtracted = ""
    for entity in LIST:
        embedded_query = perform_embedding(entity, model="text-embedding-3-small")

        # Parse embeddings if they are stored as strings
        def parse_embeddings(embeddings):
            return [json.loads(emb) if isinstance(emb, str) else emb for emb in embeddings]

        # Parse the embeddings in the dictionary
        parsed_embeddings = parse_embeddings(DICT["embedding"])
        names = DICT["name"]

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
        res = similarities[:3]

        # Print results
        print("Top 3 most similar embeddings:")
        print(f"Name: {res[0]['name']}, Similarity: {res[0]['similarity']:.4f}, Entity Type: {DICT['entity type'][names.index(res[0]['name'])]}")
        print(f"Name: {res[1]['name']}, Similarity: {res[1]['similarity']:.4f}, Entity Type: {DICT['entity type'][names.index(res[1]['name'])]}")
        print(f"Name: {res[2]['name']}, Similarity: {res[2]['similarity']:.4f}, Entity Type: {DICT['entity type'][names.index(res[2]['name'])]}")
        EntitiesExtracted = EntitiesExtracted + "Entity Name: " + res[0]['name'] + "  \nEntity type is: " + DICT['entity type'][names.index(res[0]['name'])] + " \n"
        return EntitiesExtracted

EntitiesExtracted = extract_entities(LIST_OF_ENTITY, DICT)

CYPHER = generate_cypher(USER_QUERY, PROMPT_CYPHER_QUERY_BUILDER, EntitiesExtracted)

def execute_query(query):
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            return record["paper.paper_id"]

ID_GRAPH_RESULTS = execute_query(CYPHER)

ID_VECTOR_RESULTS = list(set(perform_search_id(USER_QUERY, index_abstract)))

ID_RESULTS = ID_VECTOR_RESULTS + [ID_GRAPH_RESULTS]

def perform_search_with_filters(input_text, index, LIST_OF_IDS):
    vec = perform_embedding(input_text)  # Get the embedding vector for the input text
    query_results = index.query(
        vector=vec,  # Use the embedding vector for the search
        top_k=20,  # Return the top 2 matches
        include_values=False,
        filter={"id": {"$in": LIST_OF_IDS}},
        include_metadata=True)
    # Extract metadata text and scores from the query results
    metadata_full = [match['metadata'] for match in query_results['matches']]
    return metadata_full

FINAL_RESULTS = perform_search_with_filters(USER_QUERY, index_body, ID_RESULTS)

ATHENA = perform_response(USER_QUERY, FINAL_RESULTS, PROMPT_answer)
print(ATHENA)