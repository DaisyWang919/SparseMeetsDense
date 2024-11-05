import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import faiss
import json
import os
import sys

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download NLTK resources
nltk.download('punkt')

# Load the dev set
dev_set_path = '/local/scratch3/jnie29/REPLUG/v1.0_sample_nq-dev-sample.jsonl'
print("Loading the dev dataset...")
dev_data = pd.read_json(dev_set_path, lines=True)
print("Dev dataset loaded.")

# Load the grouped TSV file
grouped_file_path = '/local/scratch3/jnie29/REPLUG/psgs_w100_clean.tsv'
print("Loading the grouped TSV file...")
grouped_df = pd.read_csv(grouped_file_path, sep='\t')
print("Grouped TSV file loaded.")

corpus = grouped_df['text'].tolist()
# Initialize BM25 with tokenized corpus
print("Initializing BM25 with the corpus...")
with open('tokenized_content.json', 'r') as f:
    tokenized_corpus = json.load(f)
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 initialization complete.")

# Load Contriever as the dense retriever encoder and move it to the device
class ContrieverEncoder(torch.nn.Module):
    def __init__(self):
        super(ContrieverEncoder, self).__init__()
        self.question_encoder = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(device)
        self.context_encoder = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(device)
        self.question_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        self.context_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    def encode_question(self, question):
        inputs = self.question_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
        return outputs.pooler_output

    def encode_passages(self, passages):
        inputs = self.context_tokenizer(passages, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.context_encoder(**inputs)
        return outputs.pooler_output

contriever_encoder = ContrieverEncoder()
print("Contriever encoder loaded.")

# Load T5 model and tokenizer for answer generation, and move T5 model to the device
t5_model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large").to(device)
t5_tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
print("T5 model loaded.")

# FAISS with GPU support (if available)
faiss_index = None
embedding_dim = 768
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()  # Use GPU resources
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)  # FAISS index on GPU
else:
    faiss_index = faiss.IndexFlatIP(embedding_dim)  # FAISS index on CPU

# Function to embed text for queries or documents
def embed_text(text, encode_type="question"):
    if encode_type == "question":
        return contriever_encoder.encode_question(text).detach().cpu().numpy()
    elif encode_type == "passage":
        return contriever_encoder.encode_passages(text).detach().cpu().numpy()
    else:
        raise ValueError("encode_type must be 'question' or 'passage'")

# Custom retriever to use FAISS and BM25
def custom_retriever(question, top_k_sparse, top_k_dense):
    tokenized_query = word_tokenize(question.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    top_bm25_indices = [idx for idx, _ in bm25_ranked[:top_k_sparse]]
    top_docs = [corpus[idx] for idx in top_bm25_indices]

    query_embedding = embed_text(question, encode_type="question").astype(np.float32)
    doc_embeddings = embed_text(top_docs, encode_type="passage")

    embedding_dim = doc_embeddings.shape[1]
    faiss_index.add(doc_embeddings)
    D, I = faiss_index.search(query_embedding, top_k_dense)

    retrieved_passages = [top_docs[i] for i in I[0] if i < len(top_docs)]
    best_passage = retrieved_passages[0] if retrieved_passages else "No relevant passage found."

    return best_passage

# Function to generate an answer for a question using the retrieved passage
def generate_answer(question, top_k_sparse, top_k_dense):
    best_passage = custom_retriever(question, top_k_sparse, top_k_dense)

    # Generate answer using T5 with the question and the best passage as context
    input_text = f"question: {question} context: {best_passage}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to evaluate the performance across the dev dataset
def evaluate(dev_data, top_k_sparse=1000, top_k_dense=10):
    predictions = []
    references = []

    for i, row in dev_data.iterrows():
        question = row['question_text']
        true_answer = ""
        
        if row['annotations']:
            for annotation in row['annotations']:
                if 'short_answers' in annotation and annotation['short_answers']:
                    short_answer_texts = []
                    for short_answer in annotation['short_answers']:
                        if 'text' in short_answer:
                            short_answer_texts.append(short_answer['text'])
                        elif 'start_token' in short_answer and 'end_token' in short_answer:
                            start = short_answer['start_token']
                            end = short_answer['end_token']
                            document_tokens = row['document_tokens'][start:end]
                            extracted_text = ' '.join(
                                [token['token'] for token in document_tokens if 'token' in token]
                            )
                            short_answer_texts.append(extracted_text)

                    if short_answer_texts:
                        true_answer = ', '.join(short_answer_texts)
                        break

        if not true_answer:
            continue
        
        predicted_answer = generate_answer(question, top_k_sparse, top_k_dense)
        predictions.append(predicted_answer)
        references.append(true_answer)

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Reference Answer: {true_answer}")
        sys.stdout.flush()

    return predictions, references

# Calculate the evaluation metrics
def calculate_metrics(predictions, references):
    exact_matches = [1 if pred == ref else 0 for pred, ref in zip(predictions, references)]
    exact_match_score = np.mean(exact_matches)

    def token_f1(pred, ref):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        common_tokens = pred_tokens.intersection(ref_tokens)
        if not common_tokens:
            return 0.0
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    f1_scores = [token_f1(pred, ref) for pred, ref in zip(predictions, references)]
    avg_f1_score = np.mean(f1_scores)

    print(f"Exact Match Score: {exact_match_score:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")

# Run evaluation and metric calculation
predictions, references = evaluate(dev_data)
calculate_metrics(predictions, references)