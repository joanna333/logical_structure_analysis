from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
import psutil
from tqdm import tqdm

class OLMoHandler:
    def __init__(self, cache_dir="models/"):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_name = "allenai/OLMo-2-1124-7B-Instruct"
        self.cache_dir = cache_dir
        self.chat_template = "<|endoftext|><|user|>\n{}\n<|assistant|>\n"
        os.makedirs(cache_dir, exist_ok=True)
        self.logger.info(f"Initialized OLMoHandler with cache dir: {cache_dir}")
        
    def load_model(self):
        """Load model and tokenizer from cache or download"""
        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                padding_side="left"
            )
            self.logger.info("Tokenizer loaded successfully")
            
            self.logger.info("Loading model (this may take a few minutes)...")
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Remove progress_bar argument
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            mem_diff = mem_after - mem_before
            
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Memory usage: {mem_diff:.2f}MB")
            self.logger.info(f"Device map: {self.model.hf_device_map}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
            
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text from prompt with proper sampling configuration"""
        if not hasattr(self, 'model'):
            if not self.load_model():
                return ""
        
        # Format prompt with chat template        
        formatted_prompt = self.chat_template.format(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,  # Enable sampling
                temperature=0.7,  # Keep temperature for creativity
                top_p=0.95,      # Nucleus sampling
                top_k=50,        # Top-k sampling
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|endoftext|>")[0]
            )
            
            # Extract only assistant response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            assistant_response = response.split("<|assistant|>\n")[-1].split("<|endoftext|>")[0]
            return assistant_response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# # # Usage
# olmo = OLMoHandler()
# response = olmo.generate("""
#                          Generate a text explaining the relationship between different thyroid hormones.
#                          Write the text as short sentences and so, that the logical relationship between two sentences following each other can be easily classified. 
#                          Possible relationships between sentences are: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition]. 
#                          Mark the relationships between the sentences using '-[kind of relationship]-'
#                          """)

# response1 = olmo.generate("""
#                             Generate a text about the thyroid and the different thyroid hormones.
#                             Write the text as short sentences and so, that there is a logical relationship (Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition) between two sentences following each other.
#                          """)
# print(response1)
# # prompt2 = f"The text is: '{response1}'." + """
# #                             Mark the relationships between the sentences in the text using '-[kind of relationship]-'
# #                             Possible [kind of relationship]: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition].
# #                          """
# # response2 = olmo.generate(prompt2)
# # print(response2)
# prompt3 = f"The text is: '{response1}'." + """
#                                             Identify the most important keywords in this text and give a csv file as output in which keywords and their relationships with the other keywords are given.
#                                             In each row give a keyword and its relationship with the other keywords: following the format 'Keyword1, Keyword2, Relationship'.
#                                             Generate three relationships for each keyword.
#                                             Possible kind of relationship: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition].
#                                             """
# response3 = olmo.generate(prompt3)
# print(response3)