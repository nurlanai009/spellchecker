"""Utility functions for training and evaluation."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import evaluate
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, val_loss: float, step: int, 
                   checkpoint_dir: str) -> str:
    """Save model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'step': step
    }, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                   checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, 
                                                                      Optional[torch.optim.Optimizer], 
                                                                      Dict[str, Any]]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss'],
        'step': checkpoint['step']
    }
    
    return model, optimizer, metadata

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    bleu = evaluate.load("bleu")
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")
    
    results = {}
    
    # Calculate BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)
    results["bleu"] = bleu_score["bleu"]
    
    # Calculate Character Error Rate
    cer_score = cer.compute(predictions=predictions, references=references)
    results["cer"] = cer_score
    
    # Calculate Word Error Rate
    wer_score = wer.compute(predictions=predictions, references=references)
    results["wer"] = wer_score
    
    # Accuracy (percentage of exactly correct predictions)
    accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
    results["accuracy"] = accuracy
    
    return results

def get_corrected_text(outputs: torch.Tensor, tokenizer) -> List[str]:
    """Convert model outputs to corrected text."""
    # outputs: [batch_size, seq_len, vocab_size]
    predictions = outputs.argmax(2)  # [batch_size, seq_len]
    
    corrected_texts = []
    for pred in predictions:
        corrected = tokenizer.decode(pred.tolist())
        corrected_texts.append(corrected)
    
    return corrected_texts