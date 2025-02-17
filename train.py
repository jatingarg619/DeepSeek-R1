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

def create_dataloader(dataset, tokenizer, batch_size, block_size=2048, num_workers=4):
    def tokenize_function(examples):
        # Concatenate all texts and add EOS token
        text = examples['text']
        tokenized = tokenizer(
            text, 
            truncation=True,
            max_length=block_size,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create input and target sequences
        input_ids = tokenized['input_ids'][:, :-1]
        labels = tokenized['input_ids'][:, 1:]
        attention_mask = tokenized['attention_mask'][:, :-1]
        
        # Ensure proper dimensions
        input_ids = input_ids.squeeze(0)  # Remove batch dimension if present
        labels = labels.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        
        # Convert attention mask to boolean
        attention_mask = attention_mask.to(torch.bool)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        batched=True
    )
    
    return tokenized_dataset

def train(
    model: nn.Module,
    dataset,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    save_dir: str = "checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs/training", exist_ok=True)  # Create logs directory
    
    # Open log file
    log_file = open("logs/training/training_log.txt", "a")
    log_file.write("\n" + "="*50 + "\nNew Training Run\n" + "="*50 + "\n")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    global_step = 0
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    total_loss = 0
    
    # Training loop
    progress_bar = tqdm(range(config['max_steps']))
    for step in range(config['max_steps']):
        batch = next(iter(dataset))
        
        # Ensure proper tensor dimensions and types
        input_ids = batch['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
        labels = batch['labels'].unsqueeze(0).to(device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)  # Already boolean from dataloader
        
        # Forward pass with mixed precision
        with autocast(device_type=device.type):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Add auxiliary loss for load balancing if using MoE
            if hasattr(model, 'layers') and hasattr(model.layers[0], 'moe'):
                expert_counts = torch.zeros(model.config.num_experts, device=device)
                # First get the embeddings
                hidden_states = model.embed_tokens(input_ids)
                for layer in model.layers:
                    # Pass the embedded states through the gate
                    router_logits = layer.moe.gate(hidden_states)
                    router_probs = torch.softmax(router_logits, dim=-1)
                    expert_counts += router_probs.sum(dim=(0, 1))
                
                target_count = input_ids.size(0) * input_ids.size(1) / model.config.num_experts
                balance_loss = torch.mean((expert_counts - target_count).pow(2))
                aux_loss = 0.01 * balance_loss
                loss += aux_loss
            
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        total_loss += loss.item()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Update progress bar with average loss
            avg_loss = total_loss * gradient_accumulation_steps / (step + 1)
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
            progress_bar.update(1)
            
            # Print and log detailed stats every 100 steps
            if global_step % 100 == 0:
                log_message = (
                    f"\nStep {global_step}\n"
                    f"Average Loss: {avg_loss:.4f}\n"
                    f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n"
                    + "-" * 50
                )
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()  # Ensure logs are written immediately
            
            # Save checkpoint, log, and generate test outputs
            if global_step % config['save_steps'] == 0:
                checkpoint_path = os.path.join(save_dir, f'model_step_{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                checkpoint_message = f"\nCheckpoint saved: {checkpoint_path}"
                print(checkpoint_message)
                log_file.write(checkpoint_message + "\n")
                
                # Generate test outputs
                print("\nGenerating test outputs...")
                test_generation(model, tokenizer, device, log_file)
                print("Test outputs generated and logged")
                
                log_file.flush()

    # Close log file
    log_file.close()

def main():
    # Training configuration
    config = {
        'batch_size': 2,
        'gradient_accumulation_steps': 16,  # Effective batch size = 32
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_steps': 10000,  # Changed from 11000 to 10000
        'warmup_steps': 2000,
        'save_steps': 1000,
        'seed': 42
    }
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model config
    model_config = DeepSeekConfig()
    
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
    
    # Load dataset with streaming
    print("\nLoading dataset...")
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True
    )
    
    # Create training dataset
    print("\nPreparing dataset...")
    train_dataset = create_dataloader(
        dataset,
        tokenizer,
        batch_size=config['batch_size'],
        block_size=model_config.max_position_embeddings
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DeepSeekModel(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['max_steps']
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Print training config
    print("\nTraining configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("-" * 50)
    
    # Train the model
    print("\nStarting training...")
    train(
        model=model,
        dataset=train_dataset,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=config,
        save_dir='checkpoints'
    )
    print("\nTraining completed!")

if __name__ == '__main__':
    main() 