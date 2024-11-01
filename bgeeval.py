import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer
import numpy as np
import re
import faiss
import os
from sklearn.metrics import f1_score

# Load the dev set
dev_set_path = '/local/scratch3/jnie29/REPLUG/v1.0_sample_nq-dev-sample.jsonl'
dev_data = pd.read_json(dev_set_path, lines=True)

# Load passages from TSV file
psgs_file_path = '/local/scratch3/jnie29/REPLUG/psgs_w100_clean.tsv'
dataset = []
try:
    # Load the TSV dataset
    df = pd.read_csv(psgs_file_path, sep='\t')
    grouped = df.groupby('title')['text'].apply(lambda texts: ' '.join(texts)).reset_index()
    dataset = grouped.to_dict(orient='records')
except FileNotFoundError:
    print(f"File not found: {psgs_file_path}. Please check the file path.")

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
        return outputs.pooler_output

    def encode_passages(self, passages):
        inputs = self.context_tokenizer(passages, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.context_encoder(**inputs)
        return outputs.pooler_output

contriever_encoder = ContrieverEncoder()

# Load T5 model
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# FAISS index to store passage embeddings
index_file = "/local/scratch3/jnie29/REPLUG/faiss_index_bge"
embedding_dim = 768
passage_texts = []

if os.path.exists(index_file):
    print("Loading existing FAISS index from file...")
    index = faiss.read_index(index_file)
else:
    index = faiss.IndexFlatIP(embedding_dim)
    passage_embeddings = []
    for idx, passage in enumerate(dataset):
        if 'text' in passage:
            passage_text = passage['text'][:512]
            passage_texts.append(passage_text)
            embedding = contriever_encoder.encode_passages([passage_text]).detach().numpy()[0]
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            passage_embeddings.append(embedding)
            index.add(np.array([embedding]))
    faiss.write_index(index, index_file)

# Custom retriever to find relevant passage using FAISS
def custom_retriever(question, passage_texts, encoder, index, top_k=10):
    question_embedding = encoder.encode_question([question]).detach().numpy()
    question_embedding = np.ascontiguousarray(question_embedding, dtype=np.float32)
    D, I = index.search(question_embedding, top_k)

    if len(I) == 0 or len(I[0]) == 0:
        return "No relevant passage found."

    retrieved_passages = [passage_texts[i] for i in I[0] if i >= 0 and i < len(passage_texts)]
    keywords = re.findall(r'\w+', question.lower())
    passage_scores = []

    for passage in retrieved_passages:
        keyword_count = sum(1 for word in keywords if word in passage.lower())
        passage_scores.append((keyword_count, passage))

    # Sort passages by keyword count if there are any scores
    if passage_scores:
        passage_scores = sorted(passage_scores, key=lambda x: x[0], reverse=True)
        return passage_scores[0][1] if passage_scores[0][0] > 0 else "No relevant passage found."
    
    return "No relevant passage found."

# Function to generate an answer for a question
def generate_answer(question):
    passage = custom_retriever(question, passage_texts, contriever_encoder, index)
    input_text = f"Question: {question}\nContext: {passage}"
    inputs = contriever_encoder.question_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = contriever_encoder.question_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Evaluate across the entire dev dataset
def evaluate(dev_data, top_k=10):
    predictions = []
    references = []

    for i, row in dev_data.iterrows():
        question = row['question_text']
        
        # Extract the reference answer as a string or from token positions
        true_answer = ""
        if row['annotations'] and 'short_answers' in row['annotations'][0] and row['annotations'][0]['short_answers']:
            short_answer = row['annotations'][0]['short_answers'][0]
            if 'text' in short_answer:
                true_answer = short_answer['text']
            elif 'start_token' in short_answer and 'end_token' in short_answer:
                start = short_answer['start_token']
                end = short_answer['end_token']
                # Join tokens from start to end
                document_tokens = row['document_tokens'][start:end]
                true_answer = ' '.join([token['text'] for token in document_tokens if 'text' in token])
        
        # Generate model prediction
        predicted_answer = generate_answer(question)
        
        # Append the prediction and reference for evaluation
        predictions.append(predicted_answer)
        references.append(true_answer)

        # Print progress every 100 questions
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(dev_data)} questions.")

    return predictions, references

# Token-based F1 metric calculation for QA tasks
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
