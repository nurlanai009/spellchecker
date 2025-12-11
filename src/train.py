"""Training and evaluation functions with proper sequence handling and mixed precision training."""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Optional
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.utils import compute_metrics, get_corrected_text, save_checkpoint




def compute_masked_loss(output, target, criterion, pad_token_id, eos_token_id):
    """Compute loss with masking for padding and post-EOS tokens."""
    # Drop BOS
    output = output[:, 1:].contiguous()  # [B, T-1, V]
    target = target[:, 1:].contiguous()  # [B, T-1]

    # Mask for padding
    pad_mask = (target != pad_token_id)  # [B, T-1]

    # Mask out tokens after the *first* EOS in each sequence (keep up to first EOS)
    is_eos = (target == eos_token_id).int()          # [B, T-1]
    # Flag exactly the first EOS in each sequence
    is_first_eos = (is_eos.bool()) & (is_eos.cumsum(dim=1) == 1)
    first_eos_cum = is_first_eos.cumsum(dim=1)       # stays 0 before first EOS, 1 at/after
    # Positions strictly after first EOS
    after_first_eos = (first_eos_cum > 0) & (~is_first_eos)
    eos_mask = ~after_first_eos                      # keep before and at first EOS

    # Combine masks
    mask = pad_mask & eos_mask                       # [B, T-1]

    # Apply mask
    output = output[mask].view(-1, output.size(-1))  # [N, V]
    target = target[mask].view(-1)                   # [N]

    return criterion(output, target)


def get_corrected_text(output, tokenizer):
    """Convert model output to text, stopping at EOS token."""
    predictions = output.argmax(dim=-1)
    corrected_texts = []
    
    eos_token_id = tokenizer.token_to_id(tokenizer.special_tokens["eos_token"])
    
    for pred in predictions:
        # Find position of first EOS token
        eos_pos = (pred == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            # Take tokens up to first EOS
            pred = pred[:eos_pos[0]]
        
        text = tokenizer.decode(pred.tolist())
        corrected_texts.append(text)
    
    return corrected_texts

def train_epoch(
    model: nn.Module,
    iterator: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    tokenizer,
    teacher_forcing_ratio: float = 0.5,
    epoch: int = 0,
    val_fn: Callable = None,
    val_iterator: torch.utils.data.DataLoader = None,
    val_interval: int = 1000,
    save_interval: int = 5000,
    checkpoint_dir: str = 'checkpoints',
    writer: Optional[SummaryWriter] = None,
    best_val_loss: float = float('inf'),
    fp16: bool = True,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, Dict, float]:
    
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()
    
    all_refs = []
    all_preds = []
    
    # Get special token IDs
    pad_token_id = tokenizer.token_to_id(tokenizer.special_tokens["pad_token"])
    eos_token_id = tokenizer.token_to_id(tokenizer.special_tokens["eos_token"])
    
    # Ensure checkpoint directory exists
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    progress_bar = tqdm(iterator, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        step += 1
        
        # Get data
        clean_ids = batch["clean_ids"].to(device, non_blocking=True)
        noisy_ids = batch["noisy_ids"].to(device, non_blocking=True)
        noisy_lengths = batch.get("noisy_lengths")  # stays on CPU for pack_padded_sequence
        clean_texts = batch["clean_texts"]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if fp16 is enabled
        if fp16:
            with autocast():
                # Forward pass (now passing lengths so encoder can use packing)
                output, _ = model(noisy_ids, clean_ids, teacher_forcing_ratio,
                          src_lengths=noisy_lengths)
                
                # Calculate masked loss
                loss = compute_masked_loss(output, clean_ids, criterion, pad_token_id, eos_token_id)
            
            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Update parameters with gradient scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            output, _ = model(noisy_ids, clean_ids, teacher_forcing_ratio,
                  src_lengths=noisy_lengths)

            
            # Calculate masked loss
            loss = compute_masked_loss(output, clean_ids, criterion, pad_token_id, eos_token_id)
            
            # Backpropagation
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Update parameters
            optimizer.step()
        
        # Update total loss
        epoch_loss += loss.item()
        
        # Log training loss to TensorBoard
        if writer is not None:
            global_step = epoch * len(iterator) + (step - 1)
            writer.add_scalar('Loss/train', loss.item(), global_step)
        
        # Log progress
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{epoch_loss/step:.4f}",
            'time': f"{time.time() - start_time:.2f}s"
        })
        
        # Get predicted tokens and convert to text for metrics
        if step % 100 == 0:
            sample_size = min(4, clean_ids.size(0))
            sample_output = output.view(clean_ids.size(0), -1, output.shape[-1])[:sample_size]
            corrected_texts = get_corrected_text(sample_output, tokenizer)
            all_refs.extend(clean_texts[:sample_size])
            all_preds.extend(corrected_texts)
        
        # Validate periodically and save best model
        if val_fn and val_iterator and step % val_interval == 0:
            val_loss, val_metrics = val_fn(
                model, val_iterator, criterion, device, tokenizer,
                pad_token_id, eos_token_id, fp16=fp16
            )
            print(f"\nStep {step} Validation - Loss: {val_loss:.4f}, CER: {val_metrics['cer']:.4f}, WER: {val_metrics['wer']:.4f}")
            
            # Log validation metrics to TensorBoard
            if writer is not None:
                global_step = epoch * len(iterator) + (step - 1)
                writer.add_scalar('Loss/val', val_loss, global_step)
                writer.add_scalar('Metrics/CER_val', val_metrics['cer'], global_step)
                writer.add_scalar('Metrics/WER_val', val_metrics['wer'], global_step)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler': scaler.state_dict() if fp16 and scaler else None,
                }, best_checkpoint_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if step % save_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch}_step_{step}.pt'
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler': scaler.state_dict() if fp16 and scaler else None,
                }, checkpoint_path)
                print(f"Checkpoint saved at step {step}")
            
            model.train()
    
    # Compute final metrics
    metrics = compute_metrics(all_preds, all_refs) if all_refs and all_preds else {}
    
    # Compute average loss
    avg_loss = epoch_loss / len(iterator)
    
    # Log final metrics to TensorBoard
    if writer is not None:
        final_global_step = (epoch + 1) * len(iterator)
        writer.add_scalar('Loss/train_avg', avg_loss, final_global_step)
        if metrics:
            writer.add_scalar('Metrics/CER_train', metrics['cer'], final_global_step)
            writer.add_scalar('Metrics/WER_train', metrics['wer'], final_global_step)
    
    return avg_loss, metrics, best_val_loss

def evaluate(
    model: nn.Module,
    iterator: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    fp16: bool = True
) -> Tuple[float, Dict[str, float]]:
    
    model.eval()
    epoch_loss = 0
    all_refs = []
    all_preds = []
    
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            # Get data
            clean_ids = batch["clean_ids"].to(device, non_blocking=True)
            noisy_ids = batch["noisy_ids"].to(device, non_blocking=True)
            noisy_lengths = batch.get("noisy_lengths")  # CPU lengths
            clean_texts = batch["clean_texts"]
            
            # Forward pass with mixed precision if fp16 is enabled
            if fp16:
                with autocast():
                    # Forward pass (no teacher forcing during evaluation)
                    output, _ = model(noisy_ids, clean_ids, teacher_forcing_ratio=0,
                          src_lengths=noisy_lengths)
                    
                    # Calculate masked loss
                    loss = compute_masked_loss(output, clean_ids, criterion, pad_token_id, eos_token_id)
            else:
                # Forward pass (no teacher forcing during evaluation)
                output, _ = model(noisy_ids, clean_ids, teacher_forcing_ratio=0,
                      src_lengths=noisy_lengths)
                
                # Calculate masked loss
                loss = compute_masked_loss(output, clean_ids, criterion, pad_token_id, eos_token_id)
            
            epoch_loss += loss.item()
            
            # Convert predictions to text
            corrected_texts = get_corrected_text(output, tokenizer)
            all_refs.extend(clean_texts)
            all_preds.extend(corrected_texts)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_refs) if all_refs and all_preds else {}
    
    return epoch_loss / len(iterator), metrics

def predict_corrections(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    fp16: bool = True
) -> List[str]:
    
    model.eval()
    
    # Encode the texts
    encoded_texts = [tokenizer.pad_sequence(tokenizer.encode(text)) for text in texts]
    input_tensor = torch.tensor(encoded_texts, dtype=torch.long).to(device)
    
    corrected_texts = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = 16
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i+batch_size]
            
            # Create dummy target tensor
            dummy_target = torch.zeros_like(batch)
            dummy_target[:, 0] = tokenizer.token_to_id(tokenizer.special_tokens["bos_token"])
            
            # Forward pass with mixed precision if fp16 is enabled
            if fp16:
                with autocast():
                    # Forward pass
                    output, _ = model(batch, dummy_target, teacher_forcing_ratio=0)
            else:
                # Forward pass
                output, _ = model(batch, dummy_target, teacher_forcing_ratio=0)
            
            # Get corrected texts
            batch_corrected = get_corrected_text(output, tokenizer)
            corrected_texts.extend(batch_corrected)
    
    return corrected_texts