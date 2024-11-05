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

# Initialize BM25 with the tokenized corpus
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

# Load the existing FAISS index file
index_file = "/local/scratch3/jnie29/REPLUG/faiss_index_bge"
embedding_dim = 768

faiss_index = faiss.read_index(index_file)
print("FAISS index loaded successfully.")

# Function to embed text for queries or documents
def embed_text(text, encode_type="question"):
    if encode_type == "question":
        return contriever_encoder.encode_question(text).detach().cpu().numpy()
    elif encode_type == "passage":
        return contriever_encoder.encode_passages(text).detach().cpu().numpy()
    else:
        raise ValueError("encode_type must be 'question' or 'passage'")

# Function to normalize scores to a 0-1 range
def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score) if max_score != min_score else scores

# Custom hybrid retriever combining BM25 and dense retrieval using FAISS
def custom_retriever(question, top_k=10, lambda_param=0.5):
    # Step 1: Sparse retrieval using BM25
    tokenized_query = word_tokenize(question.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Step 2: Dense retrieval using FAISS
    query_embedding = embed_text(question, encode_type="question").astype(np.float32)
    D, I = faiss_index.search(query_embedding, len(corpus))  # Search across all corpus entries

    dense_scores = [D[0][i] for i in range(len(D[0]))]
    dense_indices = [I[0][i] for i in range(len(I[0]))]

    # Normalize BM25 and dense scores
    normalized_bm25_scores = normalize_scores(bm25_scores)
    normalized_dense_scores = normalize_scores(dense_scores)

    # Combine BM25 and dense scores using lambda weighting
    combined_scores = []
    for idx in range(len(normalized_bm25_scores)):
        bm25_score = normalized_bm25_scores[idx]
        if idx in dense_indices:
            dense_score = normalized_dense_scores[dense_indices.index(idx)]
        else:
            dense_score = 0  # Assign a default value if not in dense retrieval

        combined_score = lambda_param * bm25_score + (1 - lambda_param) * dense_score
        combined_scores.append((idx, combined_score))

    # Rank documents based on combined scores and return the top `k`
    ranked_documents = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    top_docs = [corpus[idx] for idx, _ in ranked_documents[:top_k]]
    
    return top_docs

# Function to generate an answer for a question using the retrieved passage
def generate_answer(question, top_k=10, lambda_param=0.5):
    top_passages = custom_retriever(question, top_k, lambda_param)
    best_passage = top_passages[0] if top_passages else "No relevant passage found."

    # Generate answer using T5 with the question and the best passage as context
    input_text = f"question: {question} context: {best_passage}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to evaluate the performance across the dev dataset
def evaluate(dev_data, top_k=10, lambda_param=0.5):
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

        predicted_answer = generate_answer(question, top_k, lambda_param)
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

# Loop through different lambda parameters and track results
for lambda_param in np.arange(0.1, 1.0, 0.1):
    print(f"\nEvaluating with lambda parameter: {lambda_param:.1f}")
    predictions, references = evaluate(dev_data, top_k=10, lambda_param=lambda_param)
    print(f"Results for lambda = {lambda_param:.1f}:")
    calculate_metrics(predictions, references)
