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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import time
from huggingface_hub import login
import atexit
import signal
import gc

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
    log_file.write("\n" + "="*50 + "\nTest Generations\n" + "="*50 + "\n")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        log_file.write(f"\nPrompt {i}: {prompt}\n")
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device
        )
        log_file.write(f"Generated text:\n{generated_text}\n")
        log_file.write("-"*50 + "\n")
    
    log_file.write("\n" + "="*50 + "\n")
    log_file.flush()

def tokenize_function(examples, tokenizer, block_size):
    text = examples['text']
    tokenized = tokenizer(
        text, 
        truncation=True,
        max_length=block_size + 1,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Get input_ids and attention_mask
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # Create input sequences and labels
    input_ids = input_ids[:, :block_size]  # Truncate to block_size
    labels = input_ids[:, 1:block_size + 1]  # Next tokens as labels
    attention_mask = attention_mask[:, :block_size]  # Match input sequence length
    
    # Create labels attention mask (shifted by 1 to match labels)
    labels_attention_mask = attention_mask[:, 1:].clone()  # Shift attention mask to match labels
    
    # Ensure proper dimensions
    input_ids = input_ids.squeeze(0)
    labels = labels.squeeze(0)
    attention_mask = attention_mask.squeeze(0)
    labels_attention_mask = labels_attention_mask.squeeze(0)
    
    # Convert attention masks to boolean
    attention_mask = attention_mask.to(torch.bool)
    labels_attention_mask = labels_attention_mask.to(torch.bool)
    
    # Replace padding token in labels with -100 using the shifted attention mask
    labels = torch.where(labels_attention_mask, labels, torch.tensor(-100))
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

def create_dataloader(dataset, tokenizer, batch_size, block_size=2048, num_workers=4):
    print("\nProcessing dataset...")
    try:
        # Create an iterator for the streaming dataset
        processed_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, block_size),
            remove_columns=dataset.column_names,
            batched=True
        )
        print("Dataset processed successfully")
        
        # Convert to iterator with buffer
        print("Creating shuffled dataset...")
        shuffled_dataset = processed_dataset.shuffle(
            seed=42,
            buffer_size=10000
        )
        print("Dataset shuffled successfully")
        
        print("Creating iterator...")
        dataset_iter = iter(shuffled_dataset)
        print("Iterator created successfully")
        
        # Verify iterator works by trying to get first batch
        try:
            print("Testing iterator...")
            first_batch = next(dataset_iter)
            print("Successfully retrieved first batch")
            print(f"Batch shapes: input_ids={first_batch['input_ids'].shape}, "
                  f"attention_mask={first_batch['attention_mask'].shape}, "
                  f"labels={first_batch['labels'].shape}")
            # Create new iterator since we consumed one batch
            dataset_iter = iter(shuffled_dataset)
            print("Reset iterator for training")
        except Exception as e:
            print(f"Error getting first batch: {str(e)}")
            raise
        
        return dataset_iter, processed_dataset
    except Exception as e:
        print(f"Error in create_dataloader: {str(e)}")
        raise

def train(
    model: nn.Module,
    dataset_iter,
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
    
    progress_bar = tqdm(range(resume_from, config['max_steps']), desc="Training")
    
    try:
        for step in range(resume_from, config['max_steps']):
            retry_count = 0
            batch = None
            
            while retry_count < config['max_retries']:
                try:
                    print(f"\nAttempting to get batch at step {step}")
                    batch = next(dataset_iter)
                    print(f"Successfully got batch with shapes: input_ids={batch['input_ids'].shape}, "
                          f"attention_mask={batch['attention_mask'].shape}, "
                          f"labels={batch['labels'].shape}")
                    break
                except StopIteration:
                    print("\nReached end of dataset, reshuffling...")
                    try:
                        # Reset iterator when we reach the end
                        shuffled_dataset = dataset.shuffle(
                            seed=step,
                            buffer_size=10000
                        ).map(
                            lambda examples: tokenize_function(examples, tokenizer, config['block_size']),
                            remove_columns=dataset.column_names,
                            batched=True
                        )
                        dataset_iter = iter(shuffled_dataset)
                        print("Successfully reset iterator")
                        continue
                    except Exception as e:
                        print(f"Error resetting iterator: {str(e)}")
                        raise
                except Exception as e:
                    retry_count += 1
                    print(f"\nError getting batch (attempt {retry_count}/{config['max_retries']}): {str(e)}")
                    if retry_count == config['max_retries']:
                        raise Exception(f"Maximum retries ({config['max_retries']}) exceeded when fetching batch")
                    
                    wait_time = config['retry_delay'] * (2 ** (retry_count - 1))
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            
            if batch is None:
                print("\nSkipping step due to None batch")
                continue
            
            try:
                print("\nMoving batch to device...")
                input_ids = batch['input_ids'].unsqueeze(0).to(device)
                labels = batch['labels'].unsqueeze(0).to(device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
                print("Batch moved to device successfully")
                
                with autocast(device_type=device.type):
                    logits = model(input_ids, attention_mask)
                    
                    if torch.isinf(logits).any():
                        continue
                    
                    try:
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    except RuntimeError:
                        continue
                    
                    if torch.isnan(loss).any():
                        continue
                    
                    if hasattr(model, 'layers') and hasattr(model.layers[0], 'moe'):
                        expert_counts = torch.zeros(model.config.num_experts, device=device)
                        hidden_states = model.embed_tokens(input_ids)
                        
                        for layer in model.layers:
                            router_logits = layer.moe.gate(hidden_states)
                            if torch.isnan(router_logits).any():
                                continue
                                
                            router_probs = torch.softmax(router_logits, dim=-1)
                            if torch.isnan(router_probs).any():
                                continue
                                
                            expert_counts += router_probs.sum(dim=(0, 1))
                        
                        target_count = input_ids.size(0) * input_ids.size(1) / model.config.num_experts
                        balance_loss = torch.mean((expert_counts - target_count).pow(2))
                        aux_loss = 0.01 * balance_loss
                        
                        if not torch.isnan(aux_loss).any():
                            loss += aux_loss
                    
                    loss = loss / gradient_accumulation_steps
                
                if torch.isnan(loss).any():
                    continue
                
                scaler.scale(loss).backward()
                
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        valid_gradients = False
                        break
                
                if not valid_gradients:
                    optimizer.zero_grad()
                    continue
                
                total_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
                    
                    if torch.isnan(grad_norm):
                        optimizer.zero_grad()
                        continue
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    avg_loss = total_loss * gradient_accumulation_steps / (step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                        "grad_norm": f"{grad_norm:.4f}"
                    })
                    progress_bar.update(1)
                    
                    if global_step % 100 == 0:
                        log_message = (
                            f"\nStep {global_step}\n"
                            f"Average Loss: {avg_loss:.4f}\n"
                            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n"
                            f"Gradient Norm: {grad_norm:.4f}\n"
                            + "-" * 50
                        )
                        log_file.write(log_message + "\n")
                        log_file.flush()
                        
                        # Save checkpoint every 100 steps
                        # Delete previous checkpoint if it exists
                        prev_step = global_step - 100
                        if prev_step > 0:
                            prev_checkpoint = os.path.join(save_dir, f'model_step_{prev_step}.pt')
                            if os.path.exists(prev_checkpoint):
                                os.remove(prev_checkpoint)
                                log_file.write(f"\nDeleted previous checkpoint: {prev_checkpoint}\n")
                        
                        # Save new checkpoint
                        checkpoint_path = os.path.join(save_dir, f'model_step_{global_step}.pt')
                        torch.save({
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': avg_loss,
                        }, checkpoint_path)
                        log_file.write(f"\nCheckpoint saved: {checkpoint_path}\n")
                        test_generation(model, tokenizer, device, log_file)
                        log_file.flush()
                    
                    if global_step % config['save_steps'] == 0:
                        # Delete previous checkpoint if it exists
                        prev_step = global_step - config['save_steps']
                        if prev_step > 0:
                            prev_checkpoint = os.path.join(save_dir, f'model_step_{prev_step}.pt')
                            if os.path.exists(prev_checkpoint):
                                os.remove(prev_checkpoint)
                                log_file.write(f"\nDeleted previous checkpoint: {prev_checkpoint}\n")
                        
                        # Save new checkpoint
                        checkpoint_path = os.path.join(save_dir, f'model_step_{global_step}.pt')
                        torch.save({
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': avg_loss,
                        }, checkpoint_path)
                        log_file.write(f"\nCheckpoint saved: {checkpoint_path}\n")
                        test_generation(model, tokenizer, device, log_file)
                        log_file.flush()

            except Exception as e:
                print(f"\nError processing batch: {str(e)}")
                continue

    except Exception as e:
        error_msg = f"\nTraining interrupted at step {global_step}: {str(e)}"
        log_file.write(error_msg + "\n")
        log_file.flush()
        
        # Ensure cleanup happens
        torch.cuda.empty_cache()
        gc.collect()
        
        raise
    
    finally:
        log_file.close()
        torch.cuda.empty_cache()
        gc.collect()

def main():
    # Cleanup function
    def cleanup():
        print("\nCleaning up resources...")
        gc.collect()
        torch.cuda.empty_cache()
    
    # Register cleanup for normal exit and signals
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signo, frame: cleanup())
    signal.signal(signal.SIGINT, lambda signo, frame: cleanup())
    
    # Training configuration
    config = {
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'learning_rate': 5e-5,
        'weight_decay': 0.1,
        'max_steps': 10000,
        'warmup_steps': 1000,
        'save_steps': 1000,
        'seed': 42,
        'max_grad_norm': 1.0,
        'resume_from': 0,
        'block_size': 2048,  # Match with model's max_position_embeddings
        'max_retries': 10,
        'retry_delay': 5,
        'timeout': 30
    }
    
    # Check for existing checkpoints
    try:
        checkpoints = []
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.startswith('model_step_') and f.endswith('.pt'):
                    try:
                        # Extract step number, handling both model_step_X.pt and other formats
                        step_num = int(''.join(filter(str.isdigit, f)))
                        checkpoints.append(step_num)
                    except ValueError:
                        continue
        
        checkpoints.sort(reverse=True)
        
        if checkpoints:
            latest_step = checkpoints[0]
            print(f"\nFound existing checkpoint at step {latest_step}")
            user_input = input("Would you like to resume from this checkpoint? (y/n): ")
            if user_input.lower() == 'y':
                config['resume_from'] = latest_step
    except Exception as e:
        print(f"\nWarning: Error checking checkpoints: {str(e)}")
        print("Starting from beginning...")
        config['resume_from'] = 0
    
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
    model_config.initializer_range = 0.01
    model_config.hidden_size = 768
    model_config.intermediate_size = 1536
    model_config.num_attention_heads = 9
    model_config.num_key_value_heads = 3
    model_config.num_hidden_layers = 30
    model_config.max_position_embeddings = config['block_size']
    model_config.num_experts = 8
    model_config.num_shared_experts = 1
    model_config.num_experts_per_token = 2
    model_config.compression_ratio = 8
    
    # Load tokenizer - Using SmolLM2 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")
    
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_token_id
    print(f"Vocabulary size: {model_config.vocab_size}")
    print(f"Padding token ID: {model_config.pad_token_id}")
    
    # Load dataset with streaming and increased timeout
    print("\nLoading Cosmopedia dataset (web_samples_v2 split)...")
    try:
        # First try to authenticate with HuggingFace
        token = input("\nPlease enter your HuggingFace token: ")
        login(token=token, add_to_git_credential=False)
        
        # Configure dataset loading with increased timeouts
        from datasets.config import HF_DATASETS_CACHE
        from datasets.utils.file_utils import get_datasets_user_agent
        from datasets.download.download_config import DownloadConfig
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Create download config
        download_config = DownloadConfig(
            token=token,
            max_retries=5,
            force_download=False,
            cache_dir=None
        )
        
        # Then load the dataset
        dataset = load_dataset(
            "HuggingFaceTB/cosmopedia",
            "web_samples_v2",
            split="train",
            streaming=True,
            download_config=download_config
        )
        print("Successfully loaded Cosmopedia dataset")
    except Exception as e:
        print(f"\nError loading Cosmopedia dataset: {str(e)}")
        raise  # Re-raise the exception since we can't proceed without the dataset
    
    print("\nDataset features:", dataset.features)
    
    # Create training dataset with reduced block size
    print("\nPreparing dataset...")
    train_dataset, dataset = create_dataloader(
        dataset,
        tokenizer,
        batch_size=config['batch_size'],
        block_size=config['block_size']  # Use smaller block size
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DeepSeekModel(model_config).to(device)
    
    # Initialize model weights with smaller range
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    model.apply(init_weights)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=True  # Enable fused optimizer operations
    )
    
    # Initialize learning rate scheduler with longer warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['max_steps']
    )
    
    # Initialize gradient scaler with more aggressive settings
    scaler = GradScaler(
        init_scale=2**10,        # Increased initial scale
        growth_factor=1.5,       # More aggressive growth
        backoff_factor=0.5,
        growth_interval=100,     # More frequent scale increases
        enabled=True
    )
    
    # Print training config
    print("\nTraining configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("-" * 50)
    
    # Train the model with resume capability
    print("\nStarting training...")
    while True:
        try:
            train(
                model=model,
                dataset_iter=train_dataset,
                dataset=dataset,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                config=config,
                save_dir='checkpoints',
                resume_from=config['resume_from']
            )
            print("\nTraining completed successfully!")
            break
        except Exception as e:
            print(f"\nTraining interrupted: {str(e)}")
            user_input = input("Would you like to resume training? (y/n): ")
            if user_input.lower() != 'y':
                print("Training terminated by user.")
                break
            
            # Find the latest checkpoint
            checkpoints = sorted([
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir('checkpoints')
                if f.startswith('model_step_') and f.endswith('.pt')
            ], reverse=True)
            
            if checkpoints:
                config['resume_from'] = checkpoints[0]
                print(f"Resuming from step {config['resume_from']}")
            else:
                print("No checkpoints found. Starting from beginning.")
                config['resume_from'] = 0

if __name__ == '__main__':
    main() 