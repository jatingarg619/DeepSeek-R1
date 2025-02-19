import torch
from model.model import DeepSeekConfig, DeepSeekModel
from transformers import AutoTokenizer
import os
import argparse
from pathlib import Path

# Test prompts
TEST_PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a short story about a time traveler who:",
    "Here's a recipe for a delicious vegetarian dish:",
    "The most fascinating discovery in astronomy was:",
    "The future of artificial intelligence will likely involve:"
]

def test_generation_from_checkpoint(checkpoint_path, device='cuda', debug=True):
    """Test generation using a saved checkpoint with detailed error handling."""
    
    print(f"\nTesting checkpoint: {checkpoint_path}")
    
    try:
        # Initialize model config
        config = DeepSeekConfig()
        config.hidden_size = 384
        config.intermediate_size = 1024
        config.num_attention_heads = 6
        config.num_key_value_heads = 2
        config.num_hidden_layers = 12
        config.max_position_embeddings = 512
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        config.vocab_size = tokenizer.vocab_size
        
        if debug:
            print(f"Vocabulary size: {tokenizer.vocab_size}")
            print(f"Padding token ID: {tokenizer.pad_token_id}")
            print(f"EOS token ID: {tokenizer.eos_token_id}")
        
        # Initialize model
        print("\nInitializing model...")
        model = DeepSeekModel(config).to(device)
        
        # Load checkpoint
        print("\nLoading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        
        if debug:
            print(f"Checkpoint step: {step}")
            print(f"Checkpoint loss: {loss}")
        
        # Set model to eval mode
        model.eval()
        
        print("\nStarting generation tests...")
        print("=" * 50)
        
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\nTesting prompt {i}: {prompt}")
            
            try:
                # Tokenize with shape logging
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                if debug:
                    print(f"Input shape: {input_ids.shape}")
                
                # Generate with error catching
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids,
                            max_length=100,
                            temperature=0.8,
                            top_k=50
                        )
                        if debug:
                            print(f"Output shape: {outputs.shape}")
                        
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print(f"\nGenerated text:\n{generated_text}")
                        
                    except RuntimeError as e:
                        print(f"Generation error: {str(e)}")
                        print(f"Input tensor shape: {input_ids.shape}")
                        continue
                        
            except Exception as e:
                print(f"Error processing prompt {i}: {str(e)}")
                continue
                
            print("-" * 50)
            
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Test text generation from a checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run generation on')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # If no checkpoint specified, use the latest one
    if not args.checkpoint:
        checkpoints_dir = Path('checkpoints')
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob('model_step_*.pt'))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[2]))
                args.checkpoint = str(latest)
                print(f"Using latest checkpoint: {args.checkpoint}")
            else:
                print("No checkpoints found in ./checkpoints/")
                return
        else:
            print("Checkpoints directory not found")
            return
    
    test_generation_from_checkpoint(args.checkpoint, args.device, args.debug)

if __name__ == '__main__':
    main() 