import pandas as pd
import numpy as np
import re
import math
import faiss
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# --- Configuration ---
chatbot_dir = "/content/drive/My Drive/Chatbot" 
csv_file_name = "Milestone QA Dataset.csv"
csv_path = os.path.join(chatbot_dir, csv_file_name)

try:
    if os.path.exists(chatbot_dir):
        os.chdir(chatbot_dir)
except Exception as e:
    print(f"Error changing directory: {e}. **FATAL: Check CHATBOT_DIR PATH.**")
    sys.exit()

embedding_model_name = 'all-MiniLM-L6-v2'
llm_model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
k_retrieval = 15
k_final = 5
chunk_size = 1024 

class MilestoneChatbot:
    def __init__(self, csv_path: str = csv_path, k_retrieval: int = k_retrieval, 
                 k_final: int = k_final, chunk_size: int = chunk_size):
        
        print("Initializing Milestone Chatbot...")
        self.csv_path = csv_path
        self.k_retrieval = k_retrieval
        self.k_final = k_final
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: List[Tuple[str, str]] = [] 

        self.df = None; self.corpus = None; self.embed_model = None; self.faiss_index = None
        self.llm_tokenizer = None; self.llm_model = None

        self._load_and_preprocess_data()
        self._setup_embedding_model()
        self._build_faiss_index()
        self._setup_llm()
        print(f"Initialization Complete. Running on device: {self.device}")

    def _text_processing(self, text: str) -> str:
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"\s+"," ", text, flags=re.I).strip()
        return text

    def _load_and_preprocess_data(self):
        try:
            df_full = pd.read_csv(self.csv_path)
            self.df = df_full[['prompt (User Question)', 'completion (Bot Answer)']].copy()
            self.df.rename(columns={'prompt (User Question)':'prompt', 'completion (Bot Answer)':'answer'}, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.df['prompt'] = self.df['prompt'].apply(self._text_processing)
            self.df['answer'] = self.df['answer'].apply(self._text_processing)
            self.corpus = self.df['answer'].tolist()
        except FileNotFoundError:
            print("CSV not found, using dummy data.")
            self.df = pd.DataFrame({'prompt': ['test'], 'answer': ['Quality is ensured through strict audits.']})
            self.corpus = self.df['answer'].tolist()

    def _setup_embedding_model(self):
        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)

    def _build_faiss_index(self):
        corpus_embeddings = self.embed_model.encode(self.corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        self.faiss_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self.faiss_index.add(corpus_embeddings)

    def _setup_llm(self):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, low_cpu_mem_usage=True).to(self.device)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.llm_model.config.pad_token_id is None:
            self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id

    def reset_history(self):
        self.history = []

    def _format_history(self) -> str:
        if not self.history: return "No previous history."
        return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.history[-2:]])

    def _post_process_answer(self, text: str, start_token: str) -> str:
        """
        Cleans placeholders and fixes perspective.
        """
        cleaned = text

        # 1. Remove [company name] placeholders
        # Replace "At [company name]" -> "We"
        cleaned = re.sub(r"at\s*\[company name\],?", "we", cleaned, flags=re.IGNORECASE)
        # Replace remaining "[company name]" -> "our company"
        cleaned = re.sub(r"\[company name\]", "our company", cleaned, flags=re.IGNORECASE)
        
        # 2. Force consistency if the model drifted
        replacements = [
            (r"\bthe company\b", "we"),
            (r"\bThe company\b", "We"),
            (r"\bthey\b", "we"),
            (r"\bThey\b", "We"),
            (r"\btheir\b", "our"),
            (r"\bTheir\b", "Our")
        ]
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned)

        # 3. Clean up common conversational filler
        bad_phrases = [
            "yes, i understand the question.",
            "in response to the user query",
            "regarding how you ensure",
            "here is some information",
            "according to the text"
        ]
        for phrase in bad_phrases:
            cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)

        # 4. Strict Start Check
        # If we forced "We" but it output "Yes, We...", remove the "Yes,"
        if start_token == "We" and cleaned.lower().strip().startswith("yes"):
            cleaned = re.sub(r"^yes,?\s*", "", cleaned.strip(), flags=re.IGNORECASE)

        # 5. Final Polish
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'^\W+', '', cleaned) # Remove leading punctuation
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            
        return cleaned

    def generate_rag_response(self, user_query: str) -> Tuple[str, float]:
        clean_query = self._text_processing(user_query)
        
        # Retrieval
        search_query = clean_query
        if self.history:
            search_query = f"{self.history[-1][0]} {clean_query}"
        
        query_embedding = self.embed_model.encode([search_query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ind = self.faiss_index.search(query_embedding, self.k_retrieval)
        retrieved_texts = [self.df.iloc[idx]['answer'] for idx in ind[0][:self.k_final]]
        context = ".\n".join(retrieved_texts)
        history_str = self._format_history()

        # --- SMART START TOKEN LOGIC ---
        start_token = "We"
        
        # If question is "Do/Are/Have/Can", start with "Yes,"
        yes_no_triggers = ["do you", "are you", "have you", "can you", "is there"]
        
        # If question is "What/Who/How/Why", strictly start with "We" or "Our"
        # We prioritize checking "What" to ensure it overrides any accidental triggers
        what_triggers = ["what", "who", "how", "why", "where"]
        
        if any(clean_query.startswith(t) for t in what_triggers):
            start_token = "We"
        elif any(trigger in clean_query for trigger in yes_no_triggers):
            start_token = "Yes,"

        # --- PROMPT ---
        prompt = f"""<|system|>
You are the Company. Rewrite the "Facts" below as "We".
1. Do not use placeholders like [company name].
2. Keep it concise.
</s>
<|user|>
Facts:
{context}

Chat History:
{history_str}

Question: {clean_query}
</s>
<|assistant|>
{start_token}"""

        input_ids = self.llm_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.chunk_size).to(self.device)

        output = self.llm_model.generate(
            input_ids['input_ids'], 
            max_new_tokens=150,        
            temperature=0.3,           
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            attention_mask=input_ids['attention_mask']
        )

        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraction
        if "<|assistant|>" in response:
            generated_part = response.split("<|assistant|>")[-1].strip()
            # Ensure the start token is preserved
            if not generated_part.lower().startswith(start_token.lower()):
                 final_answer = f"{start_token} {generated_part}"
            else:
                 final_answer = generated_part
        else:
            final_answer = response

        # --- POST PROCESSING ---
        final_answer = self._post_process_answer(final_answer, start_token)

        ppl_score = self.compute_perplexity(final_answer)
        self.history.append((clean_query, final_answer))
        return final_answer, ppl_score

    def compute_perplexity(self, text: str) -> float:
        if not text: return 0.0
        encodings = self.llm_tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        with torch.no_grad():
            loss = self.llm_model(input_ids, labels=input_ids).loss
        return math.exp(loss.item()) if loss else 0.0

if __name__ == '__main__':
    try:
        chatbot = MilestoneChatbot()
        
        test_questions = [
            "what are your measure to ensure accuracy?",
            "do you have certificates for it?",
            "who performs the audits?"
        ]

        print("\n" + "="*60)
        print("AUTOMATED BATCH RUN - FINAL FORMAT FIXES")
        print("="*60 + "\n")

        for i, q in enumerate(test_questions):
            print(f"Question {i+1}: {q}")
            ans, ppl = chatbot.generate_rag_response(q)
            print(f"Answer:   {ans}")
            print(f" PPL Score: {ppl:.4f}")
            print("-" * 60)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")