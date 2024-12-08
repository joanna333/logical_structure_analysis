from olmo_handler import OLMoHandler
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from preprocessor import TextPreprocessor

class StructureExtractor:
    _instance = None  # Singleton instance
    _model = None    # Shared model instance
    
    def __init__(self, cache_dir="models/", relationships=[]):
        if StructureExtractor._model is None:
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing OLMo model...")
            self.olmo = OLMoHandler(cache_dir=cache_dir)
            self.preprocessor = TextPreprocessor()  # Add preprocessor
            StructureExtractor._model = self.olmo
            self.relationships = [
                                    "Causal",
                                    "Conditional",
                                    "Sequential",
                                    "Comparison",
                                    "Explanation",
                                    "Definition",
                                    "Contrast",
                                    "Addition",
                                    "Emphasis",
                                    "Elaboration",
                                    "Illustration",
                                    "Concession",
                                    "Generalization",
                                    "Inference",
                                    "Summary",
                                    "Problem Solution",
                                    "Contrastive Emphasis",
                                    "Purpose",
                                    "Clarification",
                                    "Enumeration",
                                    "Cause and Effect",
                                    "Temporal Sequence",
                                ]
        else:
            self.olmo = StructureExtractor._model
            self.preprocessor = TextPreprocessor()
            
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate_text(self, topic_name: str, prompt: str = None) -> str:
        """Generate text from topic and optional prompt.
        
        Args:
            topic_name: Topic to generate text about
            prompt: Optional custom prompt
            
        Returns:
            str: Generated text
            
        Raises:
            Exception: If text generation fails
        """
        complete_prompt = f""" {prompt if prompt else topic_name}. 
                    Write the text as full sentences. The sentences should be short. The logical relationship 
                    between two sentences following each other should be easily classified. 
                    Possible relationships between sentences are: {self.relationships}. 
                    Mark the relationships between the sentences using '-[kind of relationship]-'
                """
        try:
            generated_text = self.olmo.generate(complete_prompt)
            self.logger.info(f"Generated text for topic: {topic_name}")
            return generated_text
        except Exception as e:
            self.logger.error(f"Error generating text for topic {topic_name}: {e}")
            raise

    def get_relationship(self, sentence1: str, sentence2: str) -> str:
        """Determine the relationship between two sentences.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            str: Relationship type between sentences
        """
        prompt = f"""
        Determine the relationship between the following sentences:
        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        Possible relationships: {self.relationships}
        Give only the name of the relationship between the sentences as output.
        """
        return self.olmo.generate(prompt)

    def generate_relationship_from_generated_text(self, prompt: str, topic_name: str) -> None:
        """Generate relationships between sentences from generated text.
        
        Args:
            prompt: The text prompt for generation
            topic_name: Name of the topic for file naming
            
        Raises:
            Exception: If relationship generation fails
        """
        try:
            raw_text = self.generate_text(topic_name=topic_name, prompt=prompt)
            sentences_with_labels_df = self.preprocessor.split_raw_text(raw_text)
            
            # Process generated text
            sentences = self.preprocessor.split_sentences(raw_text)
            
            # Create DataFrame
            df_text = pd.DataFrame({
                'Content': sentences
            })
            
            # Extract relationships
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
            
            # Save results
            self._save_results(df_text, relationships, topic_name, sentences_with_labels_df)
            self.logger.info(f"Generated relationships for topic: {topic_name}")
            
        except Exception as e:
            self.logger.error(f"Error processing topic {topic_name}: {e}")
            raise

    def _save_results(self, df_text: pd.DataFrame, relationships: list, topic_name: str, sentences_with_labels_df: pd.DataFrame) -> None:
        """Save generated text and relationships to CSV files."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            # Save raw text
            labeled_sentences_output_dir = os.path.join(project_root, 'data', 'processed', 'labeled_sentences')
            os.makedirs(labeled_sentences_output_dir, exist_ok=True)
            text_output = os.path.join(labeled_sentences_output_dir, f"{topic_name}_sentences_with_labels.txt")
            sentences_with_labels_df.to_csv(text_output, index=False)
            self.logger.info(f"Sentences with labels saved to: {text_output}")
            
            # Save processed results
            sentences_output_dir = os.path.join(project_root, 'data', 'processed', 'sentences')
            os.makedirs(sentences_output_dir, exist_ok=True)
            
            text_path = os.path.join(sentences_output_dir, f"{topic_name}_sentences.csv")
            df_text.to_csv(text_path, index=False)
            
            relationships_output_dir = os.path.join(project_root, 'data', 'processed', 'relationships')
            rel_path = os.path.join(relationships_output_dir, f"{topic_name}_relationships.csv")
            df_rel = pd.DataFrame(relationships)
            df_rel.to_csv(rel_path, index=False)
            
            self.logger.info(f"Results saved for topic {topic_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving results for {topic_name}: {e}")
            raise
