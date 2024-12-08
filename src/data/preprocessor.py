import re
from typing import List, Dict, Tuple
import pandas as pd


class TextPreprocessor:
    def split_raw_text(self, text: str) -> None:
        """Split text into sentences and relationships and save to CSV.
        
        Args:
            text: Input text with relationship markers
            output_file: Path to save CSV output
        """
        # Split text by new line and remove empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Extract sentences and relationships
        data = []
        for line in lines:
            # Find relationship marker if exists
            relationship_match = re.search(r'-\[(\w+)\]-?', line)
            if relationship_match:
                # Get relationship type and clean sentence
                relationship = relationship_match.group(1)
                sentence = re.sub(r'-\[\w+\]-?', '', line).strip()
                data.append({
                    'Sentence': sentence,
                    'Label': relationship
                })
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        return df

    def clean_text(self, text: str) -> str:
        """Remove markdown formatting, quotes, and relationship markers from text"""
        # Remove markdown and quotes
        cleaned_text = text.replace('**', '').replace('"', ' ')
        
        # Remove relationship markers with updated regex to match also -[Cause and Effect]-
        cleaned_text = re.sub(r'-\[[\w\s]+\]-', '', cleaned_text)  # Matches -[Definition]-, -[Causal]-, -[Cause and Effect]- etc.
        cleaned_text = re.sub(r'-\[[\w\s]+\]', '', cleaned_text)   # Matches -[Definition], -[Causal], -[Cause and Effect] etc.
        
        return cleaned_text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences and clean them"""
        cleaned_text = self.clean_text(text)
        sentences = cleaned_text.replace("\n", ". ").split(".")
        sentences = [sentence.strip() for sentence in sentences 
                    if sentence.strip() and not sentence.strip().isdigit()]
        return sentences


# Usage example:
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    with open("data/processed/content_test1/Arterial_Pulse_raw_output.txt", "r") as f:
        text = f.read()
    preprocessor.split_raw_text(text, "data/processed/content_test1/arterial_pulse_relationships.csv")
    sentences = preprocessor.split_sentences(text)
    print('Sentences', sentences)