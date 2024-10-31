from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "Brown foxes are quick and they jump high",
    "Dogs are great pets and very loyal",
    "The fox is a wild animal known for its cunning nature",
    ""
]


document_embeddings = torch.cat([embed_text(doc) for doc in documents])

query = "we keep them at home and they friendly"
query_embedding = embed_text(query)

# Compute cosine similarities between the query and each document
cosine_scores = cosine_similarity(query_embedding, document_embeddings).flatten()

# Rank documents based on similarity scores
ranked_documents = sorted(enumerate(cosine_scores), key=lambda x: x[1], reverse=True)

# Display the results
print("Dense Retrieval Ranking Results:")
for idx, score in ranked_documents:
    print(f"Document {idx+1} (Score: {score:.4f}): {documents[idx]}")
