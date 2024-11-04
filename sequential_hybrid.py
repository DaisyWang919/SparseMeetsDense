from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Sample documents (corpus)
documents = [
    """The quick brown fox jumps over the lazy dog multiple times, showcasing its agility and 
       speed in a way that makes it a remarkable creature. Such animals are often seen in forests 
       where they hunt and find food swiftly. They are known for their keen sense of smell and sharp 
       reflexes, which make them effective in their natural habitats.""",
    
    """In many ancient cultures, Emory Fodogs were revered for their loyalty and companionship. Unlike wild 
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

# Initialize BM25 with tokenized corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Define query and tokenize it
query = "fox agility and cunning nature"
tokenized_query = word_tokenize(query.lower())

# Retrieve initial set using BM25
bm25_scores = bm25.get_scores(tokenized_query)
bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
top_bm25_docs = [documents[idx] for idx, score in bm25_ranked[:3]]  # top k

# Dense retrieval setup
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Embed query and top BM25 documents
query_embedding = embed_text(query)
bm25_doc_embeddings = torch.cat([embed_text(doc) for doc in top_bm25_docs])

# Compute cosine similarities for refined ranking
cosine_scores = cosine_similarity(query_embedding, bm25_doc_embeddings).flatten()
hybrid_ranked = sorted(enumerate(cosine_scores), key=lambda x: x[1], reverse=True)

# Display results
print("Hybrid Retrieval Results:")
for idx, score in hybrid_ranked:
    print(f"Document (Score: {score:.4f}): {top_bm25_docs[idx]}")
