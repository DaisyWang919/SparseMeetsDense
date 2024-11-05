import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import json  # For storing the tokenized data as JSON for later retrieval

# Ensure you have the NLTK tokenizer resources
nltk.download('punkt')

# Read the TSV file
tsv_file_path = 'input_file.tsv'  # Replace with your actual TSV file path
df = pd.read_csv(tsv_file_path, sep='\t', header=None)  # Modify `header` if needed

# Tokenize each line and store tokens
tokenized_lines = []
for line in df[0]:  # Assuming the content is in the first column
    tokens = word_tokenize(line.lower())
    tokenized_lines.append(tokens)

# Save the tokenized data to a JSON file for later retrieval
with open('tokenized_content.json', 'w') as f:
    json.dump(tokenized_lines, f)

print("Tokenization complete. Tokenized data saved to 'tokenized_content.json'.")