# DeepSeek Architecture Implementation

This project implements the DeepSeek architecture with Multi-head Linear Attention (MLHA) and Mixture of Experts (MoE) with loss-less load balancing. The implementation is based on the original SmolLM2 architecture with significant improvements in attention mechanism and parameter efficiency.

## Key Features

1. **Multi-head Linear Attention (MLHA)**
   - Replaces traditional softmax attention with linear attention
   - Uses ELU+1 kernel for positive feature maps
   - Reduces computational complexity from O(n²) to O(n)
   - Maintains high model quality while being more efficient

2. **Mixture of Experts (MoE)**
   - Implements sparse MoE layers with 8 experts
   - Uses top-2 expert routing per token
   - Features loss-less load balancing mechanism
   - Dynamically routes tokens to the most relevant experts

3. **Load Balancing**
   - Implements auxiliary loss for balanced expert utilization
   - Uses capacity factor to prevent expert overflow
   - Ensures efficient use of all experts
   - Maintains balanced token distribution across experts

## Architecture Details

- Base model size: 576 hidden dimensions
- 30 transformer layers
- 9 attention heads
- 3 key-value heads
- 8 experts per MoE layer
- 2048 sequence length
- Uses RMSNorm for layer normalization
- Rotary positional embeddings

## Usage

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**
   ```bash
   python train.py
   ```
   The model will train on the input text file and save checkpoints in the `checkpoints` directory.

3. **Model Configuration**
   You can modify the model configuration in `model/model.py` by adjusting the `DeepSeekConfig` class parameters:
   ```python
   @dataclass
   class DeepSeekConfig:
       vocab_size: int = 49152
       hidden_size: int = 576
       num_experts: int = 8
       num_experts_per_token: int = 2
       # ... other parameters
   ```

## Implementation Details

### Multi-head Linear Attention
The implementation uses a linear attention mechanism that replaces the traditional softmax-based attention:
```python
# Linear attention computation
q = F.elu(q) + 1  # Ensure positivity
k = F.elu(k) + 1
kv = torch.matmul(k.transpose(-2, -1), v)
qkv = torch.matmul(q, kv) / seq_length
```

### Mixture of Experts
The MoE implementation includes:
- Dynamic routing based on input token features
- Load balancing through auxiliary loss
- Capacity-based token routing
- Expert selection using top-k gating

### Load Balancing
The load balancing mechanism ensures efficient expert utilization:
```python
# Calculate load balancing loss
expert_counts = torch.zeros(model.config.num_experts, device=device)
router_probs = torch.softmax(layer.moe.gate(x), dim=-1)
expert_counts += router_probs.sum(dim=(0, 1))

# Penalize uneven expert utilization
target_count = x.size(0) * x.size(1) / model.config.num_experts
balance_loss = torch.mean((expert_counts - target_count).pow(2))
```

## References

1. DeepSeek Architecture
2. "Mixture of Experts with Expert Choice Routing"
3. "Linear Transformers Are Secretly Fast Weight Memory Systems" # DeepSeek-R1
