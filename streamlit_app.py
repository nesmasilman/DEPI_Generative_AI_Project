import streamlit as st
import pandas as pd
import numpy as np
import re
import faiss
import torch
import os
import sys
import warnings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional
import math

# Disable warnings for a cleaner Streamlit app
warnings.filterwarnings('ignore')

# --- 1. Configuration Constants ---
LLM_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
EMBEDDING_MODEL_PATH = 'all-MiniLM-L6-v2'
# NOTE: Update FILE_PATH to point to your Milestone QA Dataset.csv
FILE_PATH = 'Milestone QA Dataset.csv'
K_RETRIEVAL = 5
K_FINAL = 5
CHUNK_SIZE = 512 # Defined here for initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. MilestoneChatbot Class Definition ---

class MilestoneChatbot:
    """
    RAG Chatbot using locally loaded TinyLlama for fast, single-server deployment.
    """
    def __init__(self, csv_path: str = FILE_PATH, k_retrieval: int = K_RETRIEVAL,
                 k_final: int = K_FINAL, chunk_size: int = CHUNK_SIZE): # ADDED chunk_size here
        
        self.k_retrieval = k_retrieval
        self.k_final = k_final
        self.chunk_size = chunk_size # Defined chunk_size attribute
        self.device = DEVICE

        self.df = None; self.corpus = None; self.embed_model = None; self.faiss_index = None
        self.llm_tokenizer = None; self.llm_model = None

        self._load_and_preprocess_data(csv_path)
        self._setup_embedding_model()
        self._build_faiss_index()
        self._setup_llm()

    # --- STATIC CLEANING METHODS (Simplified for consistency) ---
    # (Static methods remain unchanged)
    @staticmethod
    def _text_processing(text: str) -> str:
        if pd.isna(text): return ""
        text = str(text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"\s+"," ", text, flags=re.I).strip()
        return text

    @staticmethod
    def _remove_footnotes(text: str) -> str:
        return re.sub(r'([\)\'"])\d+([\.,]?)', r'\1\2', text)

    # --- CORE SETUP METHODS ---
    # (Setup methods remain unchanged)
    def _load_and_preprocess_data(self, csv_path: str):
        try:
            df_full = pd.read_csv(csv_path)
            self.df = df_full[['prompt (User Question)', 'completion (Bot Answer)']].copy()
            self.df.rename(columns={'prompt (User Question)':'prompt', 'completion (Bot Answer)':'answer'}, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            # Apply processing (keeping original case for final output quality)
            self.df['prompt'] = self.df['prompt'].astype(str).str.strip('"').apply(self._text_processing)
            self.df['answer'] = self.df['answer'].astype(str).str.strip('"').apply(self._text_processing).apply(self._remove_footnotes)
            self.corpus = self.df['answer'].tolist()
        except FileNotFoundError:
            st.error(f"Error: CSV file not found at {csv_path}. Using dummy data.")
            self.df = pd.DataFrame({'prompt': ['test'], 'answer': ['Quality is ensured through strict audits.']})
            self.corpus = self.df['answer'].tolist()

    def _setup_embedding_model(self):
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self.device)

    def _build_faiss_index(self):
        corpus_embeddings = self.embed_model.encode(self.corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        self.faiss_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self.faiss_index.add(corpus_embeddings)

    def _setup_llm(self):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        # Use low_cpu_mem_usage for better memory handling
        self.llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, low_cpu_mem_usage=True).to(self.device)

        # Final Tokenizer/Model configuration fixes
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.llm_model.config.pad_token_id is None:
            self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id

    # --- MAIN GENERATION METHOD ---
    # (Generation method remains unchanged, but now uses self.chunk_size correctly)
    def generate_response(self, user_query: str, history: List[Tuple[str, str]]) -> str:
        """Performs Retrieval and Generation using local LLM."""

        clean_query = self._text_processing(user_query)

        # 1. Retrieval Setup (Multi-Turn Context)
        search_query = clean_query
        if history:
            # Combine current query with previous user query for multi-turn retrieval
            search_query = f"{history[-1][0]} {clean_query}"

        query_embedding = self.embed_model.encode([search_query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ind = self.faiss_index.search(query_embedding, self.k_retrieval)

        # Get the final context (simple K=5 selection from the search results)
        retrieved_texts = [self.df.iloc[idx]['answer'] for idx in ind[0][:self.k_final]]
        context = ". ".join(retrieved_texts)

        # Format history for prompt
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-2:]])

        # 2. Prompt Construction
        prompt = f"""
        Role: Façade Engineering Expert.
        Task: Answer the User Question using the Facts below.
        Rules:
        1. Speak as "We" (the company).
        2. If the user asks a yes/no question, start with "Yes" or "No".
        3. Do not mention "the text" or "the context".

        Facts:
        {context}

        History:
        {history_str}

        User Question: {clean_query}
        Assistant:
        """

        # 3. Generation
        # Uses self.chunk_size defined in __init__
        input_ids = self.llm_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.chunk_size).to(self.device)

        output = self.llm_model.generate(
            input_ids['input_ids'],
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            attention_mask=input_ids['attention_mask']
        )

        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)

        # 4. Post-processing (Extract Assistant's answer)
        final_answer = response.split("Assistant:")[-1].strip()

        return final_answer


# --- 3. Streamlit Application Interface ---

# Use session state to cache the expensive chatbot initialization
@st.cache_resource
def initialize_chatbot():
    # Only perform the file check inside the cached function
    # NOTE: The only thing that should cause this to rerun is editing this script file itself!
    if not os.path.exists(FILE_PATH):
        st.error(f"Error: Data file not found. Please ensure '{FILE_PATH}' is in the current directory.")
        return None

    # st.spinner is critical here to show the user that work is happening
    with st.spinner("Initializing RAG Models (This may take a moment)..."):
        return MilestoneChatbot()

chatbot = initialize_chatbot()

# Set the page title and layout
st.set_page_config(page_title="Milestone Engineering Virtual Assistant", layout="centered")

# Header section
# Note: Assuming 'MileStone.jpeg' is available in the running directory
try:
    st.image("MileStone.jpeg", caption="MileStone")
except FileNotFoundError:
    st.header("MileStone Engineering")

st.title("Milestone Engineering Virtual Assistant")
st.write("Instant access to our capabilities, safety standards, and project history.")

# ... (Chat logic remains unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your inquiry here..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    if chatbot:
        with st.chat_message("assistant"):
            with st.spinner("Processing request..."):

                chat_history_for_rag = [
                    (st.session_state.messages[i]['content'], st.session_state.messages[i+1]['content'])
                    for i in range(len(st.session_state.messages) - 1)
                    if st.session_state.messages[i]['role'] == 'user' and st.session_state.messages[i+1]['role'] == 'assistant'
                ]

                response = chatbot.generate_response(prompt, chat_history_for_rag)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})