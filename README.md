# DeepSeek Architecture Implementation

This project implements a modified version of the DeepSeek architecture with Multi-head Linear Attention (MLHA) and Mixture of Experts (MoE). The implementation uses a smaller model configuration optimized for training efficiency while maintaining the core architectural features.

## Key Features

1. **Multi-head Linear Attention (MLHA)**
   - Replaces traditional softmax attention with linear attention
   - Uses ELU+1 kernel for positive feature maps
   - Reduces computational complexity from O(n²) to O(n)
   - Includes grouped query attention for efficiency

2. **Mixture of Experts (MoE)**
   - Implements sparse MoE layers with 8 experts
   - Uses top-2 expert routing per token
   - Features loss-less load balancing mechanism
   - Includes capacity-based token routing

3. **Training Features**
   - Automatic checkpoint management (saves every 500 steps)
   - Automatic resume from latest checkpoint
   - Test generations every 500 steps
   - Detailed logging with timestamps
   - Gradient scaling and clipping
   - Mixed precision training

## Model Configuration

Current training configuration:
```python
model_config = {
    'hidden_size': 384,           # Model dimension
    'intermediate_size': 1024,    # MLP dimension
    'num_attention_heads': 6,     # Number of attention heads
    'num_key_value_heads': 2,     # Number of key/value heads for grouped attention
    'num_hidden_layers': 12,      # Number of transformer layers
    'max_position_embeddings': 512 # Maximum sequence length
}

training_config = {
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'max_steps': 10000,
    'warmup_steps': 500,
    'save_steps': 500
}
```

## Usage

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**
   ```bash
   python train.py
   ```
   The model will:
   - Automatically resume from latest checkpoint if available
   - Save checkpoints every 500 steps
   - Generate test outputs for 5 prompts every 500 steps
   - Log training metrics and generations to `logs/training/`

3. **Testing Checkpoints**
   ```bash
   python test_checkpoint.py --checkpoint checkpoints/model_step_X.pt
   ```
   Tests the model's generation capabilities using 5 different prompts.

4. **Generation**
   ```bash
   python generate.py --checkpoint checkpoints/model_step_X.pt
   ```
   Generates text using saved model checkpoints with configurable parameters.

## Training Logs

The training process logs:
- Loss values and learning rates every 100 steps
- Gradient norms and timing information
- Test generations every 500 steps
- Checkpoint saving/loading events
- Dataset iteration information

Log files are saved in `logs/training/` with timestamps for easy tracking.

## Test Prompts

The model is regularly tested on these prompts during training:
1. "Explain the concept of quantum entanglement in simple terms:"
2. "Write a short story about a time traveler who:"
3. "Here's a recipe for a delicious vegetarian dish:"
4. "The most fascinating discovery in astronomy was:"
5. "The future of artificial intelligence will likely involve:"

## Implementation Details

### Training Optimizations
- Mixed precision training with gradient scaling
- Gradient accumulation (4 steps)
- Cosine learning rate schedule with warmup
- Automatic batch retry mechanism with exponential backoff
- Memory-efficient attention implementations
- Proper cleanup and resource management

### Dataset
- Uses the Cosmopedia dataset (web_samples_v2)
- Streaming mode for memory efficiency
- Automatic dataset iteration restart
- Configurable retry mechanism for network issues

## Files Structure
```
Assignment_15/
├── model/
│   └── model.py          # Model architecture implementation
├── train.py              # Training script
├── generate.py           # Generation script
├── test_checkpoint.py    # Checkpoint testing script
├── requirements.txt      # Dependencies
└── logs/
    └── training/         # Training logs with timestamps
```

## References

1. DeepSeek Architecture
2. "Mixture of Experts with Expert Choice Routing"
3. "Linear Transformers Are Secretly Fast Weight Memory Systems" # DeepSeek-R1
