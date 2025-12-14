import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Dict, Any
from torch.cuda.amp import GradScaler

from src.config import *
from src.tokenizer import Tokenizer
from src.dataset import get_dataloaders
from src.model import create_seq2seq_model
from src.utils import set_seed, count_parameters, save_checkpoint, load_checkpoint
from src.train import train_epoch, evaluate, predict_corrections

# Enable cuDNN autotuner for best LSTM/conv kernels
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def main():
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Set FP16 flag
    use_fp16 = True
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=Path("runs"))
    
    # Get train and test file paths
    # Define file paths for multiple languages and data types
    train_files = [
        "data" / "train" / "train_sentences_azeri.json",
        # "data" / "train" / "train_sentences_english.json",
        # "data" / "train" / "train_sentences_russian.json",
        "data" / "train" / "train_words_azeri.json",
        "data" / "train" / "train_social_media_azeri.json",
        "data" / "train" / "train_noised_number_sentences.json",
        # "data" / "train" / "train_words_english.json",
        # "data" / "train" / "train_words_russian.json",
        
    ]

    val_files = [
        # Testing data
        "data" / "test" / "test_sentences_azeri.json",
        "data" / "test" / "test_noised_number_sentences.json",
        # "data" / "test" / "test_sentences_english.json",
        "data" / "test" / "test_social_media_azeri.json",
        # "data" / "test" / "test_sentences_russian.json",
        "data" / "test" / "test_words_azeri.json",
        # "data" / "test" / "test_words_english.json",
        # "data" / "test" / "test_words_russian.json"
    ]

    # Verify all files exist
    for file_path in train_files + val_files:
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(max_length=MAX_LENGTH)
    
    # Check if the vocab file exists and load it
    if os.path.exists(TOKENIZER_VOCAB_PATH):
        print(f"Loading existing vocabulary from {TOKENIZER_VOCAB_PATH}")
        tokenizer.load_vocab(TOKENIZER_VOCAB_PATH)
    
    # Get data loaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        train_paths=train_files,
        val_paths=val_files,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        build_vocab=not os.path.exists(TOKENIZER_VOCAB_PATH),  # Build vocab if it doesn't exist
        vocab_save_path=TOKENIZER_VOCAB_PATH,
        truncate_train=0.5,  
        truncate_test=0.2
    )
    
    # Get vocab size from tokenizer
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model with multi-head attention
    print("Initializing Seq2Seq model with multi-head attention...")
    model = create_seq2seq_model(
        vocab_size=vocab_size,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_heads=N_HEADS,
        device=DEVICE
    )
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pad_token_id = tokenizer.token_to_id(tokenizer.special_tokens["pad_token"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=use_fp16) if use_fp16 else None
    
    # Log FP16 status
    print(f"Using FP16 mixed precision: {use_fp16}")
    
    # Specify checkpoint path (replace with actual path or set to None)
    checkpoint_path = None  # Set to None to start from scratch or specify a path
    
    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model, optimizer, metadata = load_checkpoint(model, optimizer, checkpoint_path, DEVICE)
        start_epoch = metadata['epoch'] + 1  # Resume from the next epoch
        best_val_loss = metadata['val_loss']
        
        # Load scaler state if available
        if use_fp16 and 'scaler' in metadata and metadata['scaler'] is not None:
            scaler.load_state_dict(metadata['scaler'])
    else:
        print("No checkpoint provided or found. Starting training from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
    
    # Training loop
    print(f"Starting training on {DEVICE} from epoch {start_epoch}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_metrics, best_val_loss = train_epoch(
            model=model,
            iterator=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=CLIP_GRAD,
            device=DEVICE,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
            epoch=epoch,
            val_fn=evaluate,
            val_iterator=val_loader,
            val_interval=VAL_INTERVAL,
            save_interval=SAVE_INTERVAL,
            tokenizer=tokenizer,
            checkpoint_dir=CHECKPOINT_DIR,
            writer=writer,
            best_val_loss=best_val_loss,
            fp16=use_fp16,
            scaler=scaler
        )
        
        # Evaluate after each epoch
        val_loss, val_metrics = evaluate(
            model=model,
            iterator=val_loader,
            criterion=criterion,
            device=DEVICE,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.token_to_id(tokenizer.special_tokens["eos_token"]),
            fp16=use_fp16
        )
        
        # Print metrics
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        if train_metrics:
            print(f"  Train Metrics: CER={train_metrics['cer']:.4f}, WER={train_metrics['wer']:.4f}")
        
        if val_metrics:
            print(f"  Val Metrics: CER={val_metrics['cer']:.4f}, WER={val_metrics['wer']:.4f}")
        
        # Log validation loss to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            if val_metrics:
                writer.add_scalar('Metrics/CER_val_epoch', val_metrics['cer'], epoch)
                writer.add_scalar('Metrics/WER_val_epoch', val_metrics['wer'], epoch)
        
        # Save epoch checkpoint
        checkpoint_path = Path(CHECKPOINT_DIR) / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'scaler': scaler.state_dict() if use_fp16 and scaler else None,
        }, checkpoint_path)
        print(f"Saved epoch {epoch+1} checkpoint to {checkpoint_path}")
    
    print("Training complete!")
    
    # Close TensorBoard writer
    writer.close()
    
    # Test on multi-language examples
    print("\nTesting with examples from different languages:")
    
    test_examples = {
        "azeri": [
            "salam necəsən?",
            "bakıda hava nəcədir?",
            "azərbaycan qəşəng ölkədir.",
            "mən bu dili öyrənirəm."
        ],
        # "english": [
        #     "hello how are you?",
        #     "what is the weather like in london?",
        #     "england is a beautiful country.",
        #     "i am learning this language."
        # ],
        # "russian": [
        #     "привет как дела?",
        #     "какая погода в москве?",
        #     "россия красивая страна.",
        #     "я учу этот язык."
        # ]
    }
    
    noisy_examples = {
        "azeri": [
            "salam necesen?",
            "bakida hava necedir?",
            "azerbaycan qeweng olkedir.",
            "men bu dili oyrenirem."
        ],
        # "english": [
        #     "helo how r u?",
        #     "wht is the wether lik in london?",
        #     "england is a beutiful contry.",
        #     "i am lerning this languag."
        # ],
        # "russian": [
        #     "привет как дила?",
        #     "какая пагода в москве?",
        #     "росия красивая страна.",
        #     "я учю этот езык."
        # ]
    }
    
    # Test for each language
    for lang in test_examples:
        print(f"\nTesting {lang.capitalize()} examples:")
        
        corrected_examples = predict_corrections(
            model=model,
            tokenizer=tokenizer,
            texts=noisy_examples[lang],
            device=DEVICE,
            fp16=use_fp16
        )
        
        # Print results
        for i, (original, noisy, corrected) in enumerate(zip(test_examples[lang], noisy_examples[lang], corrected_examples)):
            print(f"\nExample {i+1}:")
            print(f"  Original:  {original}")
            print(f"  Noisy:     {noisy}")
            print(f"  Corrected: {corrected}")

if __name__ == "__main__":
    main()