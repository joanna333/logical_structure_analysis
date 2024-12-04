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
        
        # Check for MPS (Apple Silicon GPU) availability
        self.device = (
            "mps" 
            if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        self.model_name = "allenai/OLMo-2-1124-7B-Instruct"
        self.cache_dir = cache_dir
        self.chat_template = "<|endoftext|><|user|>\n{}\n<|assistant|>\n"
        os.makedirs(cache_dir, exist_ok=True)
        
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
            
            self.logger.info(f"Loading model to {self.device}...")
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map={"": self.device}
            )
            
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.info(f"Model loaded. RAM usage: {mem_after - mem_before:.2f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
            
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text from prompt"""
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
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode("<|endoftext|>")[0]
        )
        
        # Extract only assistant response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_response = response.split("<|assistant|>\n")[-1].split("<|endoftext|>")[0]
        return assistant_response.strip()
    

# Initialize handler
olmo = OLMoHandler(cache_dir="models/")

# First use will download and cache
response = olmo.generate("Explain the cardiovascular system:")