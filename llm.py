import numpy as np
import openai
from dotenv import load_dotenv
import streamlit as st

class LLM:
    def __init__(self, user_query, conversation=""):
        """
        Class constructor that sets up environment variables,
        loads prompt files, initializes clients (OpenAI, DeepSeek, Neo4j, etc.),
        and stores references to the user query and indexes.
        """
        load_dotenv()

        with open("prompts/HISTORY.txt", "r") as file:
            self.HISTORY = file.read()
        
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
    
    def translate_to_english(self, query):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Traduci il testo in inglese. Non modificare il significato del testo e i suoi dettagli."},
                {"role": "user", "content": f"\n\n\n-----QUERY:{query}."}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def translate_to_italian(self, query):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Translate to italian. Do not change the meaning of the text and its details."},
                {"role": "user", "content": f"\n\n\n-----TEXT:{query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def perform_response(self, query, history):
        """Generate a final response (GPT-based) using the retrieved texts and user query."""
        response = self.client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Answer the user question. If necessary, provide additional context from conversation history"},
                {"role": "user", "content": f"\n\n-----CONVERSATION HISTORY:{history}.\n\n-----Question:{query}."}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content
        return answer
    
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
    
    def run_pipeline(self):
        self.user_query = self.translate_to_english(self.user_query)
        conversation_history = self.generate_history(self.user_query, self.conversation, self.HISTORY)
        answer = self.perform_response(self.user_query, conversation_history)
        answer = self.translate_to_italian(answer)
        return answer