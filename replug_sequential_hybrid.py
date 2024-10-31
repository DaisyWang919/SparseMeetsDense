import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer
import numpy as np
import re
import faiss
import os
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk punkt tokenizer is downloaded
nltk.download('punkt')

# Load passages from TSV file
psgs_file_path = '/local/scratch3/jnie29/REPLUG/psgs_w100_clean.tsv'
dataset = []
try:
    # Load the TSV dataset
    df = pd.read_csv(psgs_file_path, sep='\t')
    print(f"Loaded dataset with {len(df)} entries.")

    # Group text by title and create combined passages
    grouped = df.groupby('title')['text'].apply(lambda texts: ' '.join(texts)).reset_index()
    print("Dataset after grouping by title:")
    print(grouped.head(10))  # Print out a few rows of the grouped dataset to see how it looks like
    dataset = grouped.to_dict(orient='records')
except FileNotFoundError:
    print(f"File not found: {psgs_file_path}. Please check the file path.")

# Print a high-level overview of the dataset structure
if dataset:
    print("High-level structure of dataset (keys of first entry):")
    print(dataset[0].keys())

# Load Contriever as the dense retriever encoder
class ContrieverEncoder(torch.nn.Module):
    def __init__(self):
        super(ContrieverEncoder, self).__init__()
        self.question_encoder = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
        self.context_encoder = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
        self.question_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        self.context_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    def encode_question(self, question):
        inputs = self.question_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
        return outputs.pooler_output  # Shape: (batch_size, hidden_size)

    def encode_passages(self, passages):
        inputs = self.context_tokenizer(passages, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.context_encoder(**inputs)
        return outputs.pooler_output  # Shape: (batch_size, hidden_size)

contriever_encoder = ContrieverEncoder()

# Load T5 model (without its tokenizer)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
# Note: We are not loading the T5 tokenizer

# Initialize BM25 for sparse retrieval
tokenized_corpus = [word_tokenize(p['text'].lower()) for p in dataset if 'text' in p]
bm25 = BM25Okapi(tokenized_corpus)

# FAISS index to store passage embeddings
index_file = "/local/scratch3/jnie29/REPLUG/faiss_index_bge"
embedding_dim = 768  # Embedding dimension from the encoder
passage_texts = []

if os.path.exists(index_file):
    print("Loading existing FAISS index from file...")
    index = faiss.read_index(index_file)
    print("FAISS index loaded successfully.")
else:
    index = faiss.IndexFlatIP(embedding_dim)
    passage_embeddings = []

    # Precompute passage embeddings and add to the index
    for idx, passage in enumerate(dataset):
        if 'text' in passage:
            passage_text = passage['text']
            passage_text = passage_text[:512]  # Truncate passage to fit within the max length
            passage_texts.append(passage_text)
            embedding = contriever_encoder.encode_passages([passage_text]).detach().numpy()[0]
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            passage_embeddings.append(embedding)
            index.add(np.array([embedding]))
    
    # Save the FAISS index
    print("Saving FAISS index to file...")
    faiss.write_index(index, index_file)
    print("FAISS index saved successfully.")

# Hybrid retriever function
def hybrid_retriever(question, passage_texts, encoder, index, bm25, top_k=10):
    # Step 1: Sparse retrieval using BM25
    tokenized_query = word_tokenize(question.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    top_bm25_docs = [passage_texts[idx] for idx, score in bm25_ranked[:top_k]]

    # Step 2: Dense retrieval for refined ranking
    query_embedding = encoder.encode_question([question]).detach().numpy()
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
    doc_embeddings = torch.cat([encoder.encode_passages([doc]) for doc in top_bm25_docs])

    # Compute cosine similarity for ranking
    cosine_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
    hybrid_ranked = sorted(enumerate(cosine_scores), key=lambda x: x[1], reverse=True)

    # Return the highest-ranked passage based on hybrid retrieval
    return [top_bm25_docs[idx] for idx, _ in hybrid_ranked]

# Define a function to generate response for a question
def generate_answer(question):
    # Retrieve the most relevant passage using the hybrid retriever
    passages = hybrid_retriever(question, passage_texts, contriever_encoder, index, bm25)
    passage = passages[0] if passages else "No relevant passage found."
    print("Retrieved passage:", passage)

    # Combine the question and the retrieved passage for T5 input
    input_text = f"Question: {question}\nContext: {passage}"
    inputs = contriever_encoder.question_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = contriever_encoder.question_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example question
question = "where is the world s largest ice sheet located today?"
answer = generate_answer(question)
print("Answer:", answer)
