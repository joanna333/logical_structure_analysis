import os
import logging
import pandas as pd
from typing import Generator, Dict, Any
from structure_extactor import StructureExtractor
import time

class DataGenerator:
    def __init__(self, max_retries: int = 3):
        """Initialize DataGenerator with logger and extractor"""
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.data_extractor = None
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize the StructureExtractor with retries"""
        for attempt in range(self.max_retries):
            try:
                self.data_extractor = StructureExtractor()
                # Verify model is loaded
                if not hasattr(self.data_extractor.olmo, 'tokenizer'):
                    self.logger.info("Loading model...")
                    self.data_extractor.olmo.load_model()
                return
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError("Failed to initialize model after max retries")
                time.sleep(2)  # Wait before retry

    def get_topics_and_prompts(self, csv_file: str, batch_size: int = 1) -> Generator[list[Dict[str, Any]], None, None]:
        """Get topics and prompts from CSV file in batches."""
        try:
            # Get absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            file_path = os.path.join(project_root, 'data', csv_file)
            
            # Read and process CSV
            df = pd.read_csv(file_path)
            records = df.to_dict('records')
            
            for i in range(0, len(records), batch_size):
                yield records[i:i + batch_size]
                
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("CSV file is empty")
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV: {str(e)}")
            raise

    def process_topics(self, csv_file: str, batch_size: int = 1):
        """Process topics and generate relationships"""
        if not self.data_extractor:
            raise RuntimeError("StructureExtractor not initialized")
            
        try:
            for batch in self.get_topics_and_prompts(csv_file, batch_size):
                for item in batch:
                    try:
                        self.data_extractor.generate_relationship_from_generated_text(
                            prompt=item['Prompt'],
                            topic_name=item['Topic']
                        )
                        self.logger.info(f"Processed topic: {item['Topic']}")
                    except Exception as e:
                        self.logger.error(f"Error processing {item['Topic']}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run
    generator = DataGenerator()
    generator.process_topics('topics_physiology2.csv', batch_size=10)