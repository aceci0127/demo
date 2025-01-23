import os
import json
import numpy as np
import openai
from neo4j import GraphDatabase
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st

class AthenaSearch:
    def __init__(self, user_query, index_body, index_abstract, conversation=""):
        """
        Class constructor that sets up environment variables,
        loads prompt files, initializes clients (OpenAI, DeepSeek, Neo4j, etc.),
        and stores references to the user query and indexes.
        """
        load_dotenv()
        
        # Store user query and conversation
        self.user_query = user_query
        self.conversation = conversation
        
        # Environment variables
        self.OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
        self.DEEPSEEK_API_KEY = st.secrets['DEEPSEEK_API_KEY']
        self.PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        
        # OpenAI and DeepSeek clients
        self.client_openai = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        self.client_deepseek = openai.OpenAI(
            api_key=self.DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com"
        )
        
        # Pinecone client
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)

        # Pinecone indexes passed in by the user
        self.index_body = self.pc.Index(index_body)
        self.index_abstract = self.pc.Index(index_abstract)
        
        # Neo4j Setup
        AURA_DB_URI = "neo4j+s://2886f391.databases.neo4j.io"
        AURA_DB_PWD = "lrxu93e8exF48K_HTNtAi_FGrB9MOyC7J21uoNRxMaA"
        self.driver = GraphDatabase.driver(
            AURA_DB_URI, 
            auth=("neo4j", AURA_DB_PWD)
        )
        
        # Load prompt files
        with open("prompts/cypher/LLMentityExtractor.txt", "r") as file:
            self.PROMPT_LLMentity_Extractor = file.read()

        with open("prompts/cypher/CypherQueryBuilder.txt", "r") as file:
            self.PROMPT_CYPHER_QUERY_BUILDER = file.read()

        with open("prompts/ANSWER.txt", "r") as file:
            self.PROMPT_answer = file.read()
        
        with open("prompts/GENERATE_QUERY.txt", "r") as file:
            self.REGENERATE_QUERY = file.read()
        
        with open("prompts/HISTORY.txt", "r") as file:
            self.HISTORY = file.read()

    def rerank_results(self, query, docs):
        rerank_name = "cohere-rerank-3.5"
        rerank_docs = self.pc.inference.rerank(
            model=rerank_name,
            query=query,
            documents=docs,
            top_n=15,
            return_documents=True
        )
        ids = [doc["document"]["id"] for doc in rerank_docs.data]
        text = [doc["document"]["text"] for doc in rerank_docs.data]
        title = [doc["document"]["title"] for doc in rerank_docs.data]
        docs = [
            {"id": i, "text": txt, "title": tit}
            for i, txt, tit in zip(ids, text, title)
        ]
        return docs

    def generate_history(self, query, conversation, prompt):
        """Regenerate the user query based on the previous conversation and context."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\n------CONVERSATION HISTORY:{conversation}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def regenerate_query(self, query, conversation, prompt):
        """Regenerate the user query based on the previous conversation and context."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\n------CONVERSATION HISTORY:{conversation}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def generate_cypher(self, query, prompt, entities):
        """Use the DeepSeek model to generate a Cypher query based on the user query and extracted entities."""
        response = self.client_deepseek.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\nUSER QUERY{query}\n\nENTITIES EXTRACTED:{entities}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def perform_embedding(self, text, model="text-embedding-3-large"):
        """Create an embedding for the given text using OpenAI embeddings."""
        if not isinstance(text, str):
            text = str(text)
        try:
            response = self.client_openai.embeddings.create(input=text, model=model)
            vector = response.data[0].embedding
            return vector
        except Exception:
            return None

    def generate_entities(self, query, prompt):
        """Use the DeepSeek model to extract possible entities from the user query."""
        response = self.client_deepseek.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\nUSER QUERY{query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def perform_search(self, input_text, index):
        """
        Vector search on a Pinecone index.
        Returns metadata for the top matching documents.
        """
        vec = self.perform_embedding(input_text)
        query_results = index.query(
            vector=vec,
            top_k=10,
            include_values=False,
            include_metadata=True
        )
        metadata_full = [match['metadata'] for match in query_results['matches']]
        return metadata_full

    def perform_search_id(self, input_text, index):
        """Vector search on a Pinecone index, but returns only the IDs of top matches."""
        vec = self.perform_embedding(input_text)
        query_results = index.query(
            vector=vec,
            top_k=10,
            include_values=False,
            include_metadata=True
        )
        metadata_id = [match['metadata']['id'] for match in query_results['matches']]
        return metadata_id

    def perform_response(self, query, results, prompt):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\n------VECTOR RESULTS:{results}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def generate_embed_dictionary(self):
        """Pull existing embeddings from Neo4j for Materials, Topics, Applications."""
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

        with self.driver.session() as session:
            result = session.run(CYPHER_MATERIAL)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Material')

        with self.driver.session() as session:
            result = session.run(CYPHER_TOPIC)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Topic')

        with self.driver.session() as session:
            result = session.run(CYPHER_APPLICATION)
            for record in result:
                list_embeddings.append(record['embedding'])
                list_names.append(record['name'])
                list_entities.append('Application')

        # Dictionary of names, embeddings, and entity type
        return {
            "name": list_names,
            "embedding": list_embeddings,
            "entity type": list_entities
        }

    def extract_entities(self, list_of_entities, embeddings_dict):
        """
        Takes a list of entity strings and attempts to map them to the best matching
        known entity from the Neo4j embedding dictionary.
        """
        extracted = ""
        
        for entity in list_of_entities:
            # Perform embedding using smaller model for each entity
            embedded_query = self.perform_embedding(entity, model="text-embedding-3-small")

            def parse_embeddings(embeddings):
                return [
                    json.loads(emb) if isinstance(emb, str) else emb 
                    for emb in embeddings
                ]

            # Parse the embeddings stored in the dictionary
            parsed_embeddings = parse_embeddings(embeddings_dict["embedding"])
            names = embeddings_dict["name"]
            entity_types = embeddings_dict["entity type"]

            # Make sure the query vector is also parsed if it's a string
            query_vec = (
                json.loads(embedded_query) 
                if isinstance(embedded_query, str) 
                else embedded_query
            )

            # Cosine similarity function
            def cosine_similarity(vec1, vec2):
                v1, v2 = np.array(vec1), np.array(vec2)
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            similarities = []
            for name_val, emb in zip(names, parsed_embeddings):
                sim_score = cosine_similarity(emb, query_vec)
                similarities.append({"name": name_val, "similarity": sim_score})
            
            # Sort by highest similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            # Get top 3
            top3 = similarities[:3]
            # Display info (optional to keep or remove)
            print("Top 3 matches for entity:", entity)
            for i, item in enumerate(top3):
                idx_name = names.index(item['name'])
                print(f"({i+1}) Name: {item['name']}, Similarity: {item['similarity']:.4f}, Entity Type: {entity_types[idx_name]}")

            best_match_name = top3[0]['name']
            best_match_type = entity_types[names.index(best_match_name)]
            extracted += f"Entity Name: {best_match_name}\nEntity type is: {best_match_type}\n"

        return extracted

    def execute_query(self, query):
        """Executes a Cypher query in Neo4j and returns the first paper_id result found."""
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                return record["paper.paper_id"]

    def perform_search_with_filters(self, input_text, index, list_of_ids):
        """Query a Pinecone index but filter only IDs that appear in list_of_ids."""
        vec = self.perform_embedding(input_text)
        query_results = index.query(
            vector=vec,
            top_k=20,
            include_values=False,
            filter={"id": {"$in": list_of_ids}},
            include_metadata=True
        )
        return [match['metadata'] for match in query_results['matches']]

    def run_pipeline(self):
        """
        High-level method that runs the entire pipeline:
        1. Generate conversation history
        2. Regenerate user query
        3. Extract entities
        4. Generate a Cypher query
        5. Execute query in Neo4j
        6. Perform abstract-level ID search
        7. Combine IDs
        8. Filter Body index search by those IDs
        9. Rerank results
        10. Generate final response
        """
        print("User Query:", self.user_query)

        # 1. Generate conversation history
        conversation_history = self.generate_history(self.user_query, self.conversation, self.HISTORY)

        # 2. Regenerate user query
        regenerated_query = self.regenerate_query(self.user_query, conversation_history, self.REGENERATE_QUERY)
        print("Regenerated Query:", regenerated_query)

        # 3. Extract Entities
        entities_generated = self.generate_entities(
            self.user_query, 
            self.PROMPT_LLMentity_Extractor
        )
        print("Entities Extracted:", entities_generated)
        list_of_entity = [item.strip() for item in entities_generated.strip("[]").split(",")]
        print("List of Entities:", list_of_entity)
        
        # 3.1. Build a dictionary of known embeddings from Neo4j
        embeddings_dict = self.generate_embed_dictionary()
        
        # 3.2. Extract the best matching known entities from the user’s entity list
        entities_extracted = self.extract_entities(list_of_entity, embeddings_dict)
        print("Entities Extracted:", entities_extracted)
        
        # 4. Generate a Cypher query
        cypher_query = self.generate_cypher(
            regenerated_query, 
            self.PROMPT_CYPHER_QUERY_BUILDER, 
            entities_extracted
        )
        print("Cypher Query Generated:", cypher_query)
        
        # 5. Execute the Cypher query and get a single ID result from the graph
        graph_id_result = self.execute_query(cypher_query)
        print("Graph ID Result:", graph_id_result)
        
        # 6. Get top IDs from abstract index
        abstract_ids = self.perform_search_id(
            regenerated_query, 
            self.index_abstract
        )
        print("Abstract IDs:", abstract_ids)
        
        # 7. Combine the two sets of IDs (remove duplicates)
        combined_ids = list(set(abstract_ids + [graph_id_result]))
        
        # 8. Filter body index search by combined IDs
        final_results = self.perform_search_with_filters(
            regenerated_query, 
            self.index_body, 
            combined_ids
        )
        print("Final Results:", final_results)

        # 9. Rerank results
        reranked_results = self.rerank_results(regenerated_query, final_results)
        
        # 10. Generate final response
        athena_response = self.perform_response(
            regenerated_query, 
            reranked_results, 
            self.PROMPT_answer
        )
        
        return athena_response, combined_ids