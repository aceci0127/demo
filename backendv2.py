import os
import json
import numpy as np
import openai
from neo4j import GraphDatabase
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
import concurrent.futures
from google import genai

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
        self.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

        # OpenAI and DeepSeek clients
        self.client_openai = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        self.client_deepseek = openai.OpenAI(
            api_key=self.DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com"
        )
        
        self.client_gemini = genai.Client(api_key=self.GEMINI_API_KEY)
        
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
        
        with open("prompts/ANSWER2.txt", "r") as file:
            self.PROMPT_answer_2 = file.read()
        
        with open("prompts/GENERATE_QUERY.txt", "r") as file:
            self.REGENERATE_QUERY = file.read()
        
        with open("prompts/HISTORY.txt", "r") as file:
            self.HISTORY = file.read()
        
        with open("prompts/ABSTRACTLEVELQUERY.txt", "r") as file:
            self.ABSLEVQUERY = file.read()
        
        with open("prompts/TEXTLEVELQUERY.txt", "r") as file:
            self.TXTLEVQUERY = file.read()
        
        with open("prompts/SUBQUERIES.txt", "r") as file:
            self.SUBQUERIES = file.read()
        
        with open("prompts/FINALSADVANCED.txt", "r") as file:
            self.FINALSADVANCED = file.read()



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
    
    def perform_embedding_for_entity(self, text, model="text-embedding-3-small"):
        # Ensure the input text is a string
        if not isinstance(text, str):
            text = str(text)
        try:
            # Create an embedding vector using OpenAI's embedding model
            response = self.client_openai.embeddings.create(input=text, model=model)
            vector = response.data[0].embedding
            return vector
        except Exception as e:
            return None

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
    
    def regenerate_query_DEEP(self, query, prompt):
        """Regenerate the user query based on the previous conversation and context."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----USER QUERY:{query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def generate_cypher(self, query, prompt, entities):
        """Use the DeepSeek model to generate a Cypher query based on the user query and extracted entities."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
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
    
    def sub_queries(self, query, prompt):
        # Split the query into multiple parts if necessary to improve vector search
        noprompt = """
            Break down the given query into multiple logically structured sub-queries that progressively refine and explore different aspects of the main question. Ensure the sub-queries cover foundational concepts, key components, and step-by-step approaches where applicable.
            The number of subqueries should depend on the complexity of the main question and the depth of exploration required to provide a comprehensive answer.
            Don't genereate more than 4 sub-queries.
            For example:
                •	Input: How to build a RAG System?
                •	Output:
                        #	What is a RAG System?
                        #	What are the key components of a RAG System?
                        #	What are the steps to build a RAG System?”**
            
        """
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"Domanda:{query}."}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content
        # Split the answer into individual sub-queries by parsing the response
        sub_queries_list = [line.split('# ', 1)[-1] for line in answer.splitlines() if line.strip()]
        return sub_queries_list

    def generate_entities(self, query, prompt):
        """Use the DeepSeek model to extract possible entities from the user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
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
            top_k=15,
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
            top_k=20,
            include_values=False,
            include_metadata=True
        )
        metadata_id = [match['metadata']['id'] for match in query_results['matches']]
        return metadata_id
    
    def translate_to_english(self, query):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Traduci il testo in inglese. Non modificare il significato del testo e i suoi dettagli."},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def GEMINI_FUNCTION(self, prompt):
        response = self.client_gemini.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
        )
        return response.text
    
    def translate_to_italian(self):
        response = self.client_gemini.models.generate_content(
        model="gemini-2.0-flash", contents="Translate to italian. Do not change the meaning of the text and its details. Do not add any additionnal text, just the translation."
        )
        return response.text

    def perform_response(self, query, results, prompt):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----USER QUESTION:{query}\n\n------TEXT EXTRACTED:{results}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def perform_response_with_questions(self, query, results, prompt, questions):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}\n\nSUPPLEMENTARY QUESTIONS:{questions}.\n\n------VECTOR RESULTS:{results}.\n\nANSWER:"},
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

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
        3. Generate sub-queries
            4. Perform abstract-level Vector IDs search
            5. Combine IDs
            6. Filter Body index search by those IDs
            7. Rerank results
        8. Generate final response
        """
        print("User Query:", self.user_query)

        if False:
        # 0. Translate the user query to English
            self.user_query_ita = self.translate_to_english(self.user_query)

        # 1. Generate conversation history
        conversation_history = self.generate_history(self.user_query, self.conversation, self.HISTORY)

        # 2. Regenerate user query
        regenerated_query = self.regenerate_query(self.user_query, conversation_history, self.REGENERATE_QUERY)
        print("Regenerated Query:", regenerated_query)

        # 3. Generate sub-queries
        SUB_QUERIES = self.sub_queries(regenerated_query, prompt=self.SUBQUERIES)
        
        # Process sub-queries in parallel
        def process_subquery(subquery):
            # 4. Perform abstract-level Vector IDs search
            abstract_ids = self.perform_search_id(
                subquery, 
                self.index_abstract
            )
            print(f"Abstract IDs for '{subquery}':", abstract_ids)
            
            # 5. Combine the two sets of IDs (remove duplicates)
            combined_ids = list(set(abstract_ids))
            filtered_urls = [url for url in combined_ids if url is not None]
            
            # 6. Filter body index search by combined IDs
            final_results = self.perform_search_with_filters(
                subquery, 
                self.index_body, 
                filtered_urls
            )
            finalContent = "Subquery: " + subquery + "\n\nANSWER: " + final_results

            # 7. Rerank results
            reranked_results = self.rerank_results(subquery, final_results)
            return finalContent
        
        SQA = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all sub-queries to the executor and collect futures
            future_to_subquery = {executor.submit(process_subquery, subquery): subquery for subquery in SUB_QUERIES}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_subquery):
                subquery = future_to_subquery[future]
                try:
                    results = future.result()
                    SQA.append(results)
                    print(f"Completed processing for: '{subquery}'")
                except Exception as exc:
                    print(f"Sub-query '{subquery}' generated an exception: {exc}")
        
        if False:
        # Remove duplicates based on 'text'
            FINALS_UNIQUE = {}
            for entry in FINALS:
                if entry['text'] not in FINALS_UNIQUE:
                    FINALS_UNIQUE[entry['text']] = entry
            FINALS_UNIQUE = list(FINALS_UNIQUE.values())

        if False:
        # 8. Generate final response
            athena_response = self.perform_response_with_questions(
                regenerated_query, 
                FINALS_UNIQUE, 
                self.PROMPT_answer_2,
                SUB_QUERIES
                )
        
        if True:
            athena_response = self.GEMINI_FUNCTION(self.FINALSADVANCED + "\n".join(SQA))

        #11. Translate the final response to Italian
        if True:
            athena_response_ita = self.translate_to_italian(athena_response)
        
        return athena_response_ita