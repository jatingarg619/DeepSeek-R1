import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.model import DeepSeekConfig, DeepSeekModel
from typing import Optional, List
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import time
from huggingface_hub import login
import atexit
import signal
import gc
import sys

# Test prompts for generation
TEST_PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a short story about a time traveler who:",
    "Here's a recipe for a delicious vegetarian dish:",
    "The most fascinating discovery in astronomy was:",
    "The future of artificial intelligence will likely involve:"
]

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.8,
    device='cuda'
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.train()
    return generated_text

def test_generation(model, tokenizer, device, log_file):
    """Test text generation with error handling and detailed logging."""
    log_file.write("\n" + "="*50 + "\nTest Generations\n" + "="*50 + "\n")
    
    # Save model state and set to eval
    model_state = model.training
    model.eval()
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        log_file.write(f"\nPrompt {i}: {prompt}\n")
        try:
            # Tokenize with shape logging
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            log_file.write(f"Debug: Input shape: {input_ids.shape}\n")
            
            # Generate with error catching
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        input_ids,
                        max_length=100,
                        temperature=0.8,
                        top_k=50
                    )
                    log_file.write(f"Debug: Output shape: {outputs.shape}\n")
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    log_file.write(f"Generated text:\n{generated_text}\n")
                    
                except RuntimeError as e:
                    log_file.write(f"Generation error: {str(e)}\n")
                    log_file.write(f"Input tensor shape: {input_ids.shape}\n")
                    continue
                    
        except Exception as e:
            log_file.write(f"Error in prompt {i}: {str(e)}\n")
            continue
            
        log_file.write("-"*50 + "\n")
    
    # Restore model state
    model.train(model_state)
    log_file.write("\n" + "="*50 + "\n")
    log_file.flush()

def create_dataloader(dataset, tokenizer, batch_size, block_size=2048, num_workers=4):
    def tokenize_function(examples):
        text = examples['text']
        tokenized = tokenizer(
            text, 
            truncation=True,
            max_length=block_size,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Properly clone and detach tensors
        input_ids = tokenized['input_ids'].clone().detach()
        attention_mask = tokenized['attention_mask'].clone().detach()
        
        # Create labels by shifting input_ids
        labels = input_ids.clone()
        
        # Shift for next token prediction
        labels = labels[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        
        # Convert attention mask to boolean
        attention_mask = attention_mask.bool()
        
        # Create labels with -100 for masked positions
        labels = torch.where(attention_mask, labels, -100)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    # Create a batched dataset
    def batch_sampler(dataset, batch_size):
        buffer = []
        for example in dataset:
            buffer.append(example)
            if len(buffer) == batch_size:
                batch = tokenize_function({'text': [ex['text'] for ex in buffer]})
                yield batch
                buffer = []
        if buffer:
            batch = tokenize_function({'text': [ex['text'] for ex in buffer]})
            yield batch
    
    # Return an iterator that yields batches
    return batch_sampler(dataset, batch_size)

def train(
    model: nn.Module,
    dataset,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    save_dir: str = "checkpoints",
    resume_from: int = 0
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs/training", exist_ok=True)
    
    log_file = open("logs/training/training_log.txt", "a")
    log_file.write("\n" + "="*50 + "\nNew Training Run\n" + "="*50 + "\n")
    
    if resume_from > 0:
        log_file.write(f"\nResuming training from step {resume_from}\n")
        checkpoint_path = os.path.join(save_dir, f'model_step_{resume_from}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_file.write(f"Loaded checkpoint from {checkpoint_path}\n")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    global_step = resume_from
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    total_loss = 0
    accumulated_steps = 0
    
    progress_bar = tqdm(total=config['max_steps'], initial=resume_from)
    log_file.write("\nStarting training loop...\n")
    
    try:
        while global_step < config['max_steps']:
            # Create new iterator for each epoch
            data_iterator = dataset
            
            for batch in data_iterator:
                if global_step >= config['max_steps']:
                    break
                    
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Log shapes for debugging
                    log_file.write(f"Batch shapes - ids: {input_ids.shape}, labels: {labels.shape}, mask: {attention_mask.shape}\n")
                    
                    with autocast(device_type=device.type):
                        # Forward pass
                        logits = model(input_ids, attention_mask)
                        
                        # Calculate loss
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        loss = loss / gradient_accumulation_steps
                        
                        if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                            # Backward pass
                            scaler.scale(loss).backward()
                            
                            total_loss += loss.item()
                            accumulated_steps += 1
                            
                            if accumulated_steps % gradient_accumulation_steps == 0:
                                # Optimizer step
                                scaler.unscale_(optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
                                
                                if not torch.isnan(grad_norm):
                                    scaler.step(optimizer)
                                    scaler.update()
                                    scheduler.step()
                                    optimizer.zero_grad()
                                    
                                    global_step += 1
                                    
                                    # Update progress
                                    avg_loss = total_loss / accumulated_steps
                                    progress_bar.set_postfix({
                                        "loss": f"{avg_loss:.4f}",
                                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                                        "grad_norm": f"{grad_norm:.4f}"
                                    })
                                    progress_bar.update(1)
                                    
                                    # Logging
                                    if global_step % 10 == 0:
                                        log_file.write(
                                            f"\nStep {global_step} stats:\n"
                                            f"Average Loss: {avg_loss:.4f}\n"
                                            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n"
                                            f"Gradient Norm: {grad_norm:.4f}\n"
                                        )
                                        log_file.flush()
                                    
                                    # Save checkpoint if needed
                                    if global_step % 100 == 0:
                                        checkpoint_path = os.path.join(save_dir, f'model_step_{global_step}.pt')
                                        torch.save({
                                            'step': global_step,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'loss': avg_loss,
                                        }, checkpoint_path)
                                        log_file.write(f"\nSaved checkpoint: {checkpoint_path}\n")
                                        log_file.flush()
                        else:
                            log_file.write(f"Skipping step due to invalid loss: {loss.item()}\n")
                            optimizer.zero_grad()
                    
                except Exception as e:
                    log_file.write(f"Error processing batch: {str(e)}\n")
                    log_file.flush()
                    continue
        
        log_file.write("\nTraining completed successfully!\n")
        return True
        
    except Exception as e:
        error_msg = f"\nTraining interrupted: {str(e)}"
        log_file.write(error_msg + "\n")
        log_file.flush()
        return False
        
    finally:
        progress_bar.close()
        log_file.close()
        torch.cuda.empty_cache()
        gc.collect()

def cleanup(signo=None, frame=None):
    """Safe cleanup function that handles both normal exits and signals."""
    print("\nCleaning up resources...")
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Close any open files
        try:
            log_file.close()
        except:
            pass
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        # Ensure Python exits cleanly
        if signo is not None:
            sys.exit(0)

def main():
    global log_file  # Make log_file accessible to cleanup function
    
    # Register cleanup for normal exit and signals
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    # Training configuration
    config = {
        'batch_size': 16,              # Increased from 4 for better parallelization
        'gradient_accumulation_steps': 4,  # Reduced from 8 for more frequent updates
        'learning_rate': 2e-4,         # Slightly increased for faster convergence
        'weight_decay': 0.01,
        'max_steps': 10000,
        'warmup_steps': 500,
        'save_steps': 1000,
        'seed': 42,
        'max_grad_norm': 0.5,
        'resume_from': 0,
        'block_size': 512,
        'max_retries': 10,
        'retry_delay': 5,
        'timeout': 30,
        'num_workers': 4,              # Added for dataloader parallelization
        'pin_memory': True,            # Added for faster data transfer to GPU
        'prefetch_factor': 2,          # Added for data prefetching
    }
    
    # Performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix multiplications
    torch.backends.cudnn.benchmark = True         # Enable cuDNN autotuner
    torch.backends.cudnn.deterministic = False    # Disable deterministic mode for speed
    torch.set_float32_matmul_precision('high')    # Use high precision for better speed/accuracy trade-off
    
    try:
        # Check for existing checkpoints
        checkpoints = []
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                try:
                    if f.startswith('model_step_') and f.endswith('.pt'):
                        # Extract step number, handling both normal and interrupted checkpoints
                        step_str = f.replace('model_step_', '').replace('.pt', '')
                        step_str = step_str.split('_')[0]  # Remove '_interrupted' if present
                        step_num = int(step_str)
                        checkpoints.append((step_num, f))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        if checkpoints:
            latest_step, latest_file = checkpoints[0]
            print(f"\nFound existing checkpoint at step {latest_step} ({latest_file})")
            user_input = input("Would you like to resume from this checkpoint? (y/n): ")
            if user_input.lower() == 'y':
                config['resume_from'] = latest_step
        
        # Set random seed
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Enable deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # Initialize model config with adjusted parameters
        model_config = DeepSeekConfig()
        model_config.initializer_range = 0.01    # Reduced for better initialization
        model_config.hidden_size = 384
        model_config.intermediate_size = 1024
        model_config.num_attention_heads = 6
        model_config.num_key_value_heads = 2
        model_config.num_hidden_layers = 12
        model_config.max_position_embeddings = config['block_size']
        
        # Load tokenizer and set special tokens
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        
        # Ensure proper special token handling
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_config.pad_token_id = tokenizer.pad_token_id
        model_config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        model_config.eos_token_id = tokenizer.eos_token_id
        
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Padding token ID: {model_config.pad_token_id}")
        print(f"BOS token ID: {model_config.bos_token_id}")
        print(f"EOS token ID: {model_config.eos_token_id}")
        
        model_config.vocab_size = tokenizer.vocab_size
        
        # Initialize model
        print("\nInitializing model...")
        model = DeepSeekModel(model_config).to(device)
        
        # Better weight initialization
        def init_weights(module):
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for linear layers
                torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Use smaller range for embeddings
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
        model.apply(init_weights)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load dataset with streaming and improved configuration
        print("\nLoading Cosmopedia dataset...")
        try:
            # First try to authenticate with HuggingFace
            token = input("\nPlease enter your HuggingFace token: ")
            login(token=token, add_to_git_credential=False)
            
            # Configure dataset loading
            from datasets.download.download_config import DownloadConfig
            
            # Create download config with retries
            download_config = DownloadConfig(
                token=token,
                max_retries=config['max_retries'],
                force_download=False
            )
            
            # Load dataset with streaming
            dataset = load_dataset(
                "HuggingFaceTB/cosmopedia",
                "web_samples_v2",
                split="train",
                streaming=True,
                download_config=download_config
            )
            print("Successfully loaded Cosmopedia dataset")
            
            # Create training dataset
            print("\nPreparing dataset...")
            train_dataset = create_dataloader(
                dataset,
                tokenizer,
                batch_size=config['batch_size'],
                block_size=config['block_size']
            )
            print("Dataset preparation completed")
            
        except Exception as e:
            print(f"\nError loading dataset: {str(e)}")
            raise  # Re-raise the exception since we can't proceed without the dataset
        
        # Initialize optimizer with better settings
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,  # Use fused implementation
        )
        
        # Use cosine schedule with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['max_steps']
        )
        
        # Initialize gradient scaler with optimized settings
        scaler = GradScaler(
            init_scale=2**12,            # Increased from 2**10
            growth_factor=2.0,           # More aggressive growth
            backoff_factor=0.5,
            growth_interval=100
        )
        
        # Enable memory efficient attention if available
        if hasattr(model, 'enable_mem_efficient_attention'):
            model.enable_mem_efficient_attention()
        
        # Add gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Enable flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable tensor cores for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set higher precision for gradients
        torch.set_float32_matmul_precision('high')
        
        # Print training config
        print("\nTraining configuration:")
        for k, v in config.items():
            print(f"{k}: {v}")
        print("-" * 50)
        
        # Train the model with resume capability
        print("\nStarting training...")
        while True:
            try:
                training_completed = train(
                    model=model,
                    dataset=train_dataset,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=device,
                    config=config,
                    save_dir='checkpoints',
                    resume_from=config['resume_from']
                )
                
                if training_completed:
                    print("\nTraining completed successfully!")
                    break
                else:
                    print("\nTraining incomplete. Current step:", config['resume_from'])
                    user_input = input("Would you like to resume training? (y/n): ")
                    if user_input.lower() != 'y':
                        print("Training terminated by user.")
                        break
                    
                    # Find the latest checkpoint
                    checkpoints = []
                    if os.path.exists('checkpoints'):
                        for f in os.listdir('checkpoints'):
                            try:
                                if f.startswith('model_step_') and f.endswith('.pt'):
                                    step_str = f.replace('model_step_', '').replace('.pt', '')
                                    step_str = step_str.split('_')[0]
                                    step_num = int(step_str)
                                    checkpoints.append((step_num, f))
                            except (ValueError, IndexError):
                                continue
                    
                    if checkpoints:
                        latest_step = max(checkpoints, key=lambda x: x[0])[0]
                        config['resume_from'] = latest_step
                        print(f"Resuming from step {config['resume_from']}")
                    else:
                        print("No checkpoints found. Starting from beginning.")
                        config['resume_from'] = 0

            except Exception as e:
                print(f"\nTraining interrupted: {str(e)}")
                user_input = input("Would you like to resume training? (y/n): ")
                if user_input.lower() != 'y':
                    print("Training terminated by user.")
                    break
                
                # Find the latest checkpoint
                checkpoints = []
                if os.path.exists('checkpoints'):
                    for f in os.listdir('checkpoints'):
                        try:
                            if f.startswith('model_step_') and f.endswith('.pt'):
                                step_str = f.replace('model_step_', '').replace('.pt', '')
                                step_str = step_str.split('_')[0]
                                step_num = int(step_str)
                                checkpoints.append((step_num, f))
                        except (ValueError, IndexError):
                            continue
                
                if checkpoints:
                    latest_step = max(checkpoints, key=lambda x: x[0])[0]
                    config['resume_from'] = latest_step
                    print(f"Resuming from step {config['resume_from']}")
                else:
                    print("No checkpoints found. Starting from beginning.")
                    config['resume_from'] = 0

    except Exception as e:
        print(f"\nTraining interrupted: {str(e)}")
        cleanup()  # Ensure cleanup happens even on exception
        sys.exit(1)
    
    finally:
        cleanup()  # Final cleanup

if __name__ == '__main__':
    main() 