from olmo_handler import OLMoHandler
from text_to_csv import convert_text_to_csv
from preprocessor import TextPreprocessor
import pandas as pd
import os
import sys
import logging
from typing import Generator, Dict, Any
sys.path.append('..')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

DATA_PATH = r'../data/'

class StructureExtractor:
    _instance = None  # Singleton instance
    _model = None    # Shared model instance
    
    def __init__(self, cache_dir="models/"):
        if StructureExtractor._model is None:
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing OLMo model...")
            self.olmo = OLMoHandler(cache_dir=cache_dir)
            StructureExtractor._model = self.olmo
        else:
            self.olmo = StructureExtractor._model
            
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate_text(self, topic):
        prompt = f""" Generate a text about the {topic}. 
                    Write the text in the form of short sentences and so,
                    that there is a logical relationship between two sentences following each other.
                    This can be a Causal, Conditional, Sequential, Comparison, Explanation, Definition relationship.
                    Generate pure text, without sections, headlines, numbering, markdown formatting or quotes.
                    """
        return self.olmo.generate(prompt)
    
    def clean_text(self, text: str) -> str:
        """Remove markdown formatting and quotes from text"""
        
        # Remove markdown and quotes
        cleaned_text = text.replace('**', '').replace('"', ' ')
        
        return cleaned_text

    def split_sentences(self, text):
        cleaned_text = self.clean_text(text)
        sentences = cleaned_text.replace(":", ". ").replace("\n", ". ").split(".")
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() and not sentence.strip().isdigit()]
        return sentences
    
    def get_relationship(self, sentence1, sentence2):
        prompt = f"""
        Determine the relationship between the following sentences:
        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        Possible relationships: [Causal, Conditional, Sequential, Comparison, Explanation, None]
        Output: The relationship is
        """
        return self.olmo.generate(prompt)
    
    def generate_relationship_csv_from_uploaded_file(self, input_csv, output_csv="sentence_relationships.csv"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        input_path = os.path.join(project_root, 'data', 'processed', 'content', input_csv)
        output_path = os.path.join(project_root, 'data', 'processed', 'relationships', output_csv)
        
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.read_csv(input_path)
        relationships = []
        
        for index, row in df.iterrows():
            text = row['Content']
            sentences = self.split_sentences(text)
            
            for i in range(len(sentences) - 1):
                sentence1 = sentences[i]
                sentence2 = sentences[i + 1]
                relationship = self.get_relationship(sentence1, sentence2)
                
                relationships.append({
                    "Sentence 1": sentence1,
                    "Sentence 2": sentence2,
                    "Relationship": relationship
                })
        
        df_relationships = pd.DataFrame(relationships)
        df_relationships.to_csv(output_path, index=False)
        self.logger.info(f"CSV file with sentence relationships saved to {output_path}")

    def generate_relationship_csv_from_text_file(self, text_file, output_csv="sentence_relationships.csv"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        input_path = os.path.join(project_root, 'data', 'processed', 'content', text_file)
        with open(input_path, 'r') as f:
            text = f.read()
        
        sentences = self.split_sentences(text)
        relationships = []
        
        for i in range(len(sentences) - 1):
            sentence1 = sentences[i]
            sentence2 = sentences[i + 1]
            relationship = self.get_relationship(sentence1, sentence2)
            
            relationships.append({
                "Sentence 1": sentence1,
                "Sentence 2": sentence2,
                "Relationship": relationship
            })
        
        df_relationships = pd.DataFrame(relationships)
        df_relationships.to_csv(output_csv, index=False)
        self.logger.info(f"CSV file with sentence relationships saved to {output_csv}")

    def generate_relationship_from_generated_text(self, topic, topic_name):
        prompt = f"""
        Generate a text about {topic}.
        """ 
        text = self.olmo.generate(prompt)
        sentences = self.split_sentences(text)
        # save sentences to csv-file
        df_text = pd.DataFrame({
            'Content': sentences
        })
        

        relationships = []
        
        for i in range(len(sentences) - 1):
            sentence1 = sentences[i]
            sentence2 = sentences[i + 1]
            relationship = self.get_relationship(sentence1, sentence2)
            
            relationships.append({
                "Sentence 1": sentence1,
                "Sentence 2": sentence2,
                "Relationship": relationship
            })
        
        df_relationships = pd.DataFrame(relationships)
        topic = topic.replace(" ", "_")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        text_output = os.path.join(project_root, 'data', 'processed', 'content', f"{topic_name}.txt")
        df_text.to_csv(text_output, index=False)
        output_csv=f"{topic_name}_sentence_relationships.csv"
        output_path = os.path.join(project_root, 'data', 'processed', 'relationships',  output_csv)
        df_relationships.to_csv(output_path, index=False)
        self.logger.info(f"CSV file with sentence relationships saved to {output_path}")

def get_topics_and_prompts(csv_file: str, batch_size: int = 1) -> Generator[list[Dict[str, Any]], None, None]:
    """
    Get topics and prompts from CSV file in batches.
    
    Args:
        csv_file: Path to the CSV file containing topics and prompts
        batch_size: Number of records to yield in each batch
        
    Yields:
        List of dictionaries containing topic and prompt data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
    """
    try:
        df = pd.read_csv(csv_file)
        records = df.to_dict('records')
        
        for i in range(0, len(records), batch_size):
            yield records[i:i + batch_size]
            
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {csv_file}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("CSV file is empty")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

# Update main
if __name__ == "__main__":
    extractor = StructureExtractor()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
        
    topics_path = os.path.join(project_root, 'data', 'topics_sample.csv')
    # Process topics in batches
    for batch in get_topics_and_prompts(topics_path, batch_size=1):
        for item in batch:
            try:
                extractor.generate_relationship_from_generated_text(
                    topic=item['Prompt'],
                    topic_name=item['Topic']
                )
                extractor.logger.info(f"Processed topic: {item['Topic']}")
            except Exception as e:
                extractor.logger.error(f"Error processing {item['Topic']}: {e}")