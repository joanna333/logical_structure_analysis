import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

def convert_text_to_csv(input_file, output_file):
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Read text file
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sentence_id': range(len(sentences)),
        'content': sentences
    })
    
    # Save as CSV
    df.to_csv(output_file, index=False)
    print(f"Converted {len(sentences)} sentences to {output_file}")

# Usage
