from olmo_handler import OLMoHandler
import pandas as pd

class StructureExtractor:
    def __init__(self, cache_dir="models/"):
        self.olmo = OLMoHandler(cache_dir=cache_dir)
        
    def split_sentences(self, text):
        sentences = text.split(". ")
        sentences = [sentence.strip() + "." if not sentence.endswith('.') else sentence.strip() for sentence in sentences]
        return sentences
    
    def get_relationship(self, sentence1, sentence2):
        prompt = f"""
        Determine the relationship between the following sentences:
        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        Possible relationships: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, None]
        Output: The relationship is
        """
        return self.olmo.generate(prompt)
    
    def generate_relationship_csv_from_uploaded_file(self, input_csv, output_csv="sentence_relationships.csv"):
        df = pd.read_csv(input_csv)
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
        df_relationships.to_csv(output_csv, index=False)
        print(f"CSV file with sentence relationships saved to {output_csv}")


input_csv = "../data/content/summarized_Abdomen.csv"  # Replace with the path to your uploaded CSV file
output_csv = "../data/relationships/summarized_Abdomen_sentence_relationships.csv"
structure_extractor = StructureExtractor()
structure_extractor.generate_relationship_csv_from_uploaded_file(input_csv, "sentence_relationships.csv")


# # Initialize handler
# olmo = OLMoHandler(cache_dir="models/")

# # First use will download and cache
# response = olmo.generate("Determine the relationship between the following sentences: Sentence 1: 'The abdomen is the front part of the torso between the thorax chest and pelvis in humans and in other vertebrates.' Sentence 2: 'The area occupied by the abdomen is called the abdominal cavity.' Possible relationships: [Causal, Conditional, Sequential, Comparison, Contradiction] Output: The relationship is")
# print(response)