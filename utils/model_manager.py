import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple

class ModelManager:
    """Manages the transformer model for chatbot"""
    
    def __init__(self, model_dir: str = "training/results/best_model"):
        """Initialize ModelManager
        
        Args:
            model_dir (str): Directory containing the model files
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
    async def load_model(self) -> bool:
        """Load the model and tokenizer
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                local_files_only=True
            ).to(self.device)
            
            self.model.eval()
            logging.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            return False
            
    async def generate_response(
        self,
        prompt: str,
        max_length: int = 256,
        min_length: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 2
    ) -> Tuple[str, float]:
        """Generate response from the model
        
        Args:
            prompt (str): Input text
            max_length (int): Maximum length of generated text
            min_length (int): Minimum length of generated text
            temperature (float): Sampling temperature
            top_k (int): Top k sampling parameter
            top_p (float): Top p sampling parameter
            num_beams (int): Number of beams for beam search
            no_repeat_ngram_size (int): Size of n-grams to prevent repetition
            
        Returns:
            Tuple[str, float]: Generated text and confidence score
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            response = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )
            
            # Calculate confidence score
            confidence = torch.mean(torch.stack(outputs.scores)).item()
            
            return response, confidence
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "", 0.0
            
    def cleanup(self):
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()