import os
import sys
sys.path.append(os.getcwd())
root_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), r'/src/data'))
sys.path.append(os.path.join(os.getcwd(), r'/src/models'))
sys.path.append("..")
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import LabelEncoder
from src.models.BiLSTM_01 import BiLSTMAttentionBERT
import numpy as np
from torch.serialization import add_safe_globals
import pandas as pd

def get_device():
    if torch.backends.mps.is_available():
        try:
            # Test MPS availability
            torch.zeros(1).to('mps')
            return torch.device('mps')
        except:
            print("Warning: MPS device found but unavailable, falling back to CPU")
            return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_model_for_prediction(model_path):
    # Add numpy dtypes and scalar types to safe globals
    add_safe_globals([
        np.float64, np.int64, np.bool_,
        np.dtype, np._core.multiarray.scalar,
        np.dtypes.Float64DType
    ])
    
    # Force CPU
    device = torch.device('cpu')
    torch.backends.mps.enabled = False
    
    try:
        # First try loading with weights_only
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True: {str(e)}")
        print("Attempting to load without weights_only...")
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False  # Fallback
        )
    
    try:
        # Initialize model
        model = BiLSTMAttentionBERT(
            hidden_dim=128,
            num_classes=22,
            num_layers=2,
            dropout=0.5
        ).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize label encoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(checkpoint.get('label_encoder_classes', 
                        ['Addition', 'Causal', 'Cause and Effect', 'Clarification', 'Comparison',
                         'Concession', 'Conditional', 'Contrast', 'Contrastive Emphasis',
                          'Definition', 'Elaboration', 'Emphasis', 'Enumeration', 'Explanation', 
                          'Generalization', 'Illustration', 'Inference', 'Problem Solution', 'Purpose', 
                          'Sequential', 'Summary', 'Temporal Sequence']))
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'dmis-lab/biobert-base-cased-v1.2',
            local_files_only=True
        )
        
        return model, label_encoder, tokenizer
        
    except Exception as e:
        print(f"Error initializing model components: {str(e)}")
        return None, None, None

def predict_sentence(model, sentence, tokenizer, label_encoder, device=None):
    """
    Make prediction for a single sentence with label validation.
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        sentence,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    try:
        with torch.no_grad():
            # Get model outputs
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and probability
            prob, pred_idx = torch.max(probabilities, dim=1)
            
            # Validate prediction index
            if pred_idx.item() >= len(label_encoder.classes_):
                print(f"Warning: Model predicted invalid label index {pred_idx.item()}")
                return "Unknown", 0.0
                
            # Convert to label
            try:
                predicted_class = label_encoder.classes_[pred_idx.item()]
                return predicted_class, prob.item()
            except IndexError:
                print(f"Warning: Invalid label index {pred_idx.item()}")
                return "Unknown", 0.0
                
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error", 0.0
    
def print_labels(label_encoder, show_counts=False):
    """Print all labels and their corresponding indices"""
    print("\nAvailable labels:")
    print("-" * 40)
    for idx, label in enumerate(label_encoder.classes_):
        print(f"Index {idx}: {label}")
    print("-" * 40)
    print(f"Total number of classes: {len(label_encoder.classes_)}\n")

def predict_sentence2(sentence, model, tokenizer, label_encoder):
    # Tokenize the input
    inputs = tokenizer(sentence, 
                      padding=True,
                      truncation=True,
                      return_tensors='pt',
                      max_length=512)
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    # Convert prediction to label
    predicted_label = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    
    return predicted_label


if __name__ == '__main__':
    model_path = 'models/model_epoch8_acc72.53.pt'
    try:
        model, label_encoder, tokenizer = load_model_for_prediction(model_path)
        
        if all((model, label_encoder, tokenizer)):
            print_labels(label_encoder)
            df = pd.read_csv('data/raw/history_01.csv')
            for sentence in df['Sentence']:
                #sentence = "Across cultures and eras, common goals in medicine, such as curing illness and prolonging life, can be observed."
                label, confidence = predict_sentence(model, sentence, tokenizer, label_encoder)
                
                if label not in ("Unknown", "Error"):
                    
                    print(f"Predicted: {label} (confidence: {confidence:.2f})")
                else:
                    print(f"Prediction failed: {label}")
    except Exception as e:
        print(f"Error: {str(e)}")
