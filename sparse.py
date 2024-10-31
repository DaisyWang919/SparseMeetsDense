from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk


# nltk.download('punkt_tab')

documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "Brown foxes are quick and they jump high",
    "Dogs are great pets and very loyal",
    "The fox is a wild animal known for its cunning nature"
]


tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]

bm25 = BM25Okapi(tokenized_corpus)
query = "quick fox dogs"
tokenized_query = word_tokenize(query.lower())

# Get BM25 scores for the query against each document
scores = bm25.get_scores(tokenized_query)

# Sort documents by score in descending order
ranked_documents = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print(ranked_documents)

print("Documents based on scores: ")
for idx, score in ranked_documents:
    print(f"Document {idx+1} (Score: {score:.4f}): {documents[idx]}")