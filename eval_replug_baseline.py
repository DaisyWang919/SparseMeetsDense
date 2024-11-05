import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer
import numpy as np
import re
import faiss
import os
from sklearn.metrics import f1_score
import sys

# Load the dev set
dev_set_path = '/local/scratch3/jnie29/REPLUG/v1.0_sample_nq-dev-sample.jsonl'
dev_data = pd.read_json(dev_set_path, lines=True)

# Load the grouped TSV file
grouped_file_path = '/local/scratch3/jnie29/REPLUG/psgs_w100_grouped.tsv'
grouped_df = pd.read_csv(grouped_file_path, sep='\t')

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

print('loading model')
sys.stdout.flush()
# Load T5 model and tokenizer specifically for T5
t5_model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")

# FAISS index to store passage embeddings
index_file = "/local/scratch3/jnie29/REPLUG/faiss_index_bge"
embedding_dim = 768

if os.path.exists(index_file):
    print("Loading existing FAISS index from file...")
    index = faiss.read_index(index_file)
else:
    index = faiss.IndexFlatIP(embedding_dim)
    for idx, row in grouped_df.iterrows():
        passage_text = row['text'][:512]  # Truncate text if needed
        embedding = contriever_encoder.encode_passages([passage_text]).detach().numpy()[0]
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        index.add(np.array([embedding]))
    faiss.write_index(index, index_file)

# Custom retriever to find relevant passage using FAISS
def custom_retriever(question, grouped_df, encoder, index, top_k=10):
    # Encode the question to get the dense representation
    question_embedding = encoder.encode_question([question]).detach().numpy()
    question_embedding = np.ascontiguousarray(question_embedding, dtype=np.float32)

    # Use FAISS to retrieve top-k passage indices based on similarity
    D, I = index.search(question_embedding, top_k)
    
    print(f"Indices Retrieved: {I}")
    
    if len(I) == 0 or len(I[0]) == 0:
        return "No relevant passage found.", [], []

    # Retrieve the top passages using the indices from the grouped DataFrame
    retrieved_passages = []
    retrieved_embeddings = []
    
    for i in I[0]:
        if i >= 0 and i < len(grouped_df):
            passage_text = grouped_df.iloc[i]['text']
            retrieved_passages.append(passage_text)

            # Cast `i` to int before passing it to `reconstruct()`
            passage_embedding = index.reconstruct(int(i))
            retrieved_embeddings.append(passage_embedding)

    # Optional: Rank retrieved passages by keyword matching (for additional filtering)
    keywords = re.findall(r'\w+', question.lower())
    passage_scores = []

    for idx, passage in enumerate(retrieved_passages):
        keyword_count = sum(1 for word in keywords if word in passage.lower())
        passage_scores.append((keyword_count, passage, retrieved_embeddings[idx]))

    # Sort passages by keyword count if there are any scores
    if passage_scores:
        passage_scores = sorted(passage_scores, key=lambda x: x[0], reverse=True)
        best_passage = passage_scores[0][1] if passage_scores[0][0] > 0 else "No relevant passage found."
        best_embedding = passage_scores[0][2] if passage_scores[0][0] > 0 else None
    else:
        best_passage = "No relevant passage found."
        best_embedding = None

    # Return the best passage and embedding, as well as all retrieved passages and embeddings
    return best_passage, best_embedding, retrieved_passages, retrieved_embeddings

# Function to generate an answer for a question
def generate_answer(question):
    # Get the best passage, its embedding, and all retrieved passages with embeddings
    best_passage, best_embedding, retrieved_passages, retrieved_embeddings = custom_retriever(
        question, grouped_df, contriever_encoder, index
    )
    
    print(f"\nQuestion: {question}")
    print(f"Selected Passage: {best_passage}")
    
    # Generate answer using the best passage and the T5 tokenizer
    input_text = f"question: {question} context: {best_passage}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    outputs = t5_model.generate(**inputs, num_return_sequences=1, num_beams=2, max_new_tokens=50)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to evaluate the performance across the dev dataset
def evaluate(dev_data, top_k=10):
    predictions = []
    references = []

    for i, row in dev_data.iterrows():
        question = row['question_text']

        # Extract the reference answer from annotations
        true_answer = ""
        if row['annotations']:
            # Iterate over annotations to find short answers
            for annotation in row['annotations']:
                if 'short_answers' in annotation and annotation['short_answers']:
                    # Handle cases with text or token positions
                    short_answer_texts = []
                    for short_answer in annotation['short_answers']:
                        if 'text' in short_answer:
                            short_answer_texts.append(short_answer['text'])
                        elif 'start_token' in short_answer and 'end_token' in short_answer:
                            start = short_answer['start_token']
                            end = short_answer['end_token']
                            # Extract tokens between start and end
                            document_tokens = row['document_tokens'][start:end]
                            extracted_text = ' '.join(
                                [token['token'] for token in document_tokens if 'token' in token]
                            )
                            short_answer_texts.append(extracted_text)

                    if short_answer_texts:
                        true_answer = ', '.join(short_answer_texts)
                        break  # Stop after finding the first valid annotation

        # Generate model prediction
        predicted_answer = generate_answer(question)

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Reference Answer: {true_answer}")
        sys.stdout.flush()
        if not true_answer:
            continue  # Skip if no valid true answer is found

        predictions.append(predicted_answer)
        references.append(true_answer)

        if i % 100 == 0:
            print(f"Processed {i+1}/{len(dev_data)} questions.")

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
