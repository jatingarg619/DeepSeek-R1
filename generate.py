import torch
from model.model import DeepSeekConfig, DeepSeekModel
from transformers import AutoTokenizer
import random
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    device='cuda'
):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate with better sampling
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits from the model
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            probs = torch.softmax(top_k_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            nucleus_mask = cumulative_probs < top_p
            nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
            nucleus_mask[..., 0] = True
            
            # Filter logits and indices
            filtered_logits = top_k_logits[nucleus_mask]
            filtered_indices = top_k_indices[nucleus_mask]
            
            # Sample from filtered distribution
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = filtered_indices[torch.multinomial(probs, 1)]
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained DeepSeek model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize model config
    config = DeepSeekConfig()
    
    # Load tokenizer - Using SmolLM2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = DeepSeekModel(config).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Example prompts for generation
    prompts = [
        "Explain the concept of quantum entanglement in simple terms:",
        "Write a short story about a time traveler who:",
        "Here's a recipe for a delicious vegetarian dish:",
        "The most fascinating discovery in astronomy was:",
        "The future of artificial intelligence will likely involve:"
    ]
    
    print("\n" + "="*50 + "\nGenerating 5 different outputs:\n" + "="*50 + "\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGeneration {i}:")
        print(f"Prompt: {prompt}\n")
        
        # Generate with different temperatures for variety
        temperature = 0.8 + (i * 0.1)  # Vary temperature between 0.8 and 1.2
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            max_length=200,
            device=args.device
        )
        
        print(f"Generated text:\n{generated_text}\n")
        print("-"*50)

if __name__ == '__main__':
    main() 