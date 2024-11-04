from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
import torch


nltk.download('punkt')

# Initialize corpus with longer sample documents
documents = [
    """The quick brown fox jumps over the lazy dog multiple times, showcasing its agility and 
       speed in a way that makes it a remarkable creature. Such animals are often seen in forests 
       where they hunt and find food swiftly. They are known for their keen sense of smell and sharp 
       reflexes, which make them effective in their natural habitats.""",
    
    """In many ancient cultures, dogs were revered for their loyalty and companionship. Unlike wild 
       animals, domesticated dogs serve various roles for humans, including guarding, herding, and 
       assisting people with disabilities. Their bond with humans is unparalleled, making them a 
       preferred choice for pet owners worldwide.""",
    
    """Foxes are fascinating animals, often depicted in folklore and stories due to their cunning 
       nature. They are generally solitary animals, preferring to stay hidden from larger predators. 
       Their adaptability allows them to thrive in various environments, from forests to urban areas. 
       Despite their small size, foxes are quick and agile.""",
    
    """The study of animal behavior has revealed insights into the intelligence of various species. 
       For instance, many animals display problem-solving skills and memory capabilities that were 
       previously thought to be unique to humans. Animals like foxes and dogs can learn patterns 
       and adapt to changes in their environment, which showcases their cognitive abilities.""",
    
    """Wild animals often have to fend for themselves, relying on their instincts and learned 
       behaviors to survive. Predators such as lions, tigers, and even foxes demonstrate hunting 
       strategies that increase their chances of capturing prey. Survival in the wild demands both 
       physical prowess and a keen understanding of the environment."""
]

# Tokenize the documents for BM25
tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# query here
query = "fox agility and cunning nature"
tokenized_query = word_tokenize(query.lower())

# Step 1: Retrieve scores from BM25
bm25_scores = bm25.get_scores(tokenized_query)

# Step 2: Retrieve scores from dense retrieval using BAAI/bge-large-en-v1.5
# Load model and tokenizer
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Embed query and all documents
query_embedding = embed_text(query)
document_embeddings = torch.cat([embed_text(doc) for doc in documents])

# Compute cosine similarities for dense scores
cosine_scores = cosine_similarity(query_embedding, document_embeddings).flatten()

# Step 3: Combine BM25 and Dense scores with lambda weighting
lambda_param = 0.5  # Adjust this value to tune between BM25 and dense retrieval
combined_scores = [(idx, lambda_param * bm25_score + (1 - lambda_param) * dense_score) 
                   for idx, (bm25_score, dense_score) in enumerate(zip(bm25_scores, cosine_scores))]

# Rank documents based on combined scores
ranked_documents = sorted(combined_scores, key=lambda x: x[1], reverse=True)

print("Weighted Hybrid Retrieval Results:")
for idx, score in ranked_documents:
    print(f"Document {idx+1} (Score: {score:.4f}): {documents[idx]}")
