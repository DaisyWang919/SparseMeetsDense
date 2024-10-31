import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModel, AutoTokenizer
import numpy as np
import re
import faiss
import os

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
        #print(f"Question embedding shape: {outputs.pooler_output.shape}")  # Debug print
        return outputs.pooler_output  # Shape: (batch_size, hidden_size)

    def encode_passages(self, passages):
        inputs = self.context_tokenizer(passages, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.context_encoder(**inputs)
        #print(f"Passage embedding shape: {outputs.pooler_output.shape}")  # Debug print
        return outputs.pooler_output  # Shape: (batch_size, hidden_size)

contriever_encoder = ContrieverEncoder()

# Load T5 model (without its tokenizer)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
# Note: We are not loading the T5 tokenizer

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
    # Using enumerate to keep track of the iteration count
    for idx, passage in enumerate(dataset):
        if 'text' in passage:
            passage_text = passage['text']
            passage_text = passage_text[:512]  # Truncate passage to fit within the max length
            passage_texts.append(passage_text)
            embedding = contriever_encoder.encode_passages([passage_text]).detach().numpy()[0]
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            
            # Only print every 1000 iterations
            if idx % 1000 == 0:
                print(f"Adding embedding to FAISS index for passage {idx}: {passage_text[:50]}...")
            
            passage_embeddings.append(embedding)
            index.add(np.array([embedding]))
    
    # Save the FAISS index
    print("Saving FAISS index to file...")
    faiss.write_index(index, index_file)
    print("FAISS index saved successfully.")

# Custom retriever to find the most relevant passage using FAISS
def custom_retriever(question, passage_texts, encoder, index, top_k=10):
    # Encode the question
    question_embedding = encoder.encode_question([question]).detach().numpy()
    question_embedding = np.ascontiguousarray(question_embedding, dtype=np.float32)
    print(f"Question embedding for retrieval: {question_embedding}")  # Debug print
    # Search in FAISS index for similar passages
    D, I = index.search(question_embedding, top_k)
    print(f"FAISS search distances: {D}, indices: {I}")  # Debug print
    if len(I) == 0 or len(I[0]) == 0:
        return "No relevant passage found."
    retrieved_passages = [passage_texts[i] for i in I[0] if i >= 0 and i < len(passage_texts)]
    # If no valid passages are found, return a default response
    if len(retrieved_passages) == 0:
        return "No relevant passage found."
    # Further filtering: prioritize passages that contain keywords from the question
    keywords = re.findall(r'\w+', question.lower())
    passage_scores = []
    for passage in retrieved_passages:
        keyword_count = sum(1 for word in keywords if word in passage.lower())
        passage_scores.append((keyword_count, passage))
    # Sort passages by keyword count
    passage_scores = sorted(passage_scores, key=lambda x: x[0], reverse=True)
    return passage_scores[0][1] if passage_scores[0][0] > 0 else "No relevant passage found."  # Return the passage with the highest score

# Define a function to generate response for a question
def generate_answer(question):
    # Retrieve the most relevant passage using the custom retriever
    passage = custom_retriever(question, passage_texts, contriever_encoder, index)
    # Debug print to check the retrieved passage
    print("Retrieved passage:", passage)
    # Combine the question and the retrieved passage for T5 input
    input_text = f"Question: {question}\nContext: {passage}"
    print("Input to T5 model:", input_text)
    # Use the BGE tokenizer instead of the T5 tokenizer
    inputs = contriever_encoder.question_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Generate the answer using the T5 model with specified max_new_tokens
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = contriever_encoder.question_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example question
question = "where is the world s largest ice sheet located today?"
answer = generate_answer(question)
print("Answer:", answer)
