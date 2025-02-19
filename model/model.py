import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DeepSeekConfig:
    vocab_size: int = 49152
    hidden_size: int = 768
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    eos_token_id: int = 0
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    num_experts: int = 8
    num_shared_experts: int = 1
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    compression_ratio: int = 8  # For MLHA

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_rope_frequencies(dim: int, max_position_embeddings: int, theta: float = 10000.0):
    position = torch.arange(max_position_embeddings).unsqueeze(1)
    div_term = theta ** (torch.arange(0, dim, 2).float() / dim)
    freqs = position / div_term
    return freqs

def apply_rotary_embeddings(x: torch.Tensor, freqs: torch.Tensor):
    x_rot = x.float()
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x1, x2 = x_rot[..., :x_rot.shape[-1]//2], x_rot[..., x_rot.shape[-1]//2:]
    cos = torch.cos(freqs).to(x.device)
    sin = torch.sin(freqs).to(x.device)
    cos = cos.expand_as(x1)
    sin = sin.expand_as(x1)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    return torch.cat([x1_rot, x2_rot], dim=-1).to(x.dtype)

class MultiheadLinearAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Ensure latent_dim is consistent
        self.latent_dim = (self.head_dim + config.compression_ratio - 1) // config.compression_ratio
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        # Latent projections with consistent dimensions
        self.q_latent = nn.Linear(self.head_dim, self.latent_dim, bias=False)
        self.k_latent = nn.Linear(self.head_dim, self.latent_dim, bias=False)
        self.v_latent = nn.Linear(self.head_dim, self.latent_dim, bias=False)
        
        # Output projections
        self.lat_out_proj = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Layer norm for latent space
        self.latent_norm = RMSNorm(self.latent_dim)
        
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                self.head_dim,
                config.max_position_embeddings,
                config.rope_theta
            ),
            persistent=False
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [batch, seq, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        
        # Reshape and handle grouped query attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_embeddings(q, self.rope_freqs[:seq_length])
        k = apply_rotary_embeddings(k, self.rope_freqs[:seq_length])
        
        # Handle grouped query attention - repeat KV heads
        if self.num_kv_heads < self.num_heads:
            # Compute repeat factor
            repeat_factor = self.num_heads // self.num_kv_heads
            # Repeat k and v for each query head
            k = k.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Project to latent space - ensure consistent dimensions
        q = q.view(-1, self.head_dim)  # [(batch * seq * heads), head_dim]
        k = k.view(-1, self.head_dim)  # [(batch * seq * heads), head_dim]
        v = v.view(-1, self.head_dim)  # [(batch * seq * heads), head_dim]
        
        # Apply latent projections
        q_latent = self.q_latent(q)  # [(batch * seq * heads), latent_dim]
        k_latent = self.k_latent(k)  # [(batch * seq * heads), latent_dim]
        v_latent = self.v_latent(v)  # [(batch * seq * heads), latent_dim]
        
        # Reshape back to [batch, seq, heads, latent_dim]
        q_latent = q_latent.view(batch_size, seq_length, self.num_heads, self.latent_dim)
        k_latent = k_latent.view(batch_size, seq_length, self.num_heads, self.latent_dim)
        v_latent = v_latent.view(batch_size, seq_length, self.num_heads, self.latent_dim)
        
        # Apply layer norm in latent space
        q_latent = self.latent_norm(q_latent)
        k_latent = self.latent_norm(k_latent)
        v_latent = self.latent_norm(v_latent)
        
        # Rearrange for attention computation
        q_latent = q_latent.transpose(1, 2)  # [batch, heads, seq, latent_dim]
        k_latent = k_latent.transpose(1, 2)  # [batch, heads, seq, latent_dim]
        v_latent = v_latent.transpose(1, 2)  # [batch, heads, seq, latent_dim]
        
        # Linear attention in latent space
        q_latent = F.elu(q_latent) + 1
        k_latent = F.elu(k_latent) + 1
        
        # Compute attention in linear space
        kv = torch.matmul(k_latent.transpose(-2, -1), v_latent)  # [batch, heads, latent_dim, latent_dim]
        qkv = torch.matmul(q_latent, kv)  # [batch, heads, seq, latent_dim]
        
        # Scale by sequence length for stability
        qkv = qkv / seq_length
        
        # Handle attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, seq_length, 1)
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            qkv = qkv.masked_fill(~attention_mask, 0.0)
        
        # Project back to head_dim
        qkv = qkv.transpose(1, 2).contiguous()  # [batch, seq, heads, latent_dim]
        qkv = qkv.view(-1, self.latent_dim)  # [(batch * seq * heads), latent_dim]
        qkv = self.lat_out_proj(qkv)  # [(batch * seq * heads), head_dim]
        qkv = qkv.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Final output projection
        qkv = qkv.view(batch_size, seq_length, self.num_heads * self.head_dim)
        return self.o_proj(qkv)

class ExpertMLP(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MixtureOfExperts(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.expert_capacity = int(config.expert_capacity_factor * 
                                 (config.max_position_embeddings / config.num_experts))
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(config.num_experts)])

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Gate computation
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]
        
        # Top-k expert selection
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Process tokens through experts
        final_output = torch.zeros_like(hidden_states)
        capacity = self.expert_capacity
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq]
            if not expert_mask.any():
                continue
                
            expert_input = hidden_states[expert_mask]
            if len(expert_input) > capacity:
                # Apply load balancing: select only 'capacity' tokens based on importance scores
                importance_scores = routing_weights[..., expert_idx][expert_mask]  # [num_tokens]
                indices = expert_mask.nonzero(as_tuple=False)  # [num_tokens, 2]
                top_indices = torch.topk(importance_scores, k=capacity).indices  # indices within the selected tokens
                selected_indices = indices[top_indices]  # [capacity, 2]
                new_expert_mask = torch.zeros_like(expert_mask, dtype=torch.bool)
                new_expert_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
                expert_mask = new_expert_mask
                expert_input = hidden_states[expert_mask]
                
            # Process tokens through expert
            expert_output = self.experts[expert_idx](expert_input)
            final_output[expert_mask] += expert_output
            
        return final_output

class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.self_attn = MultiheadLinearAttention(config)
        self.moe = MixtureOfExperts(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones((input_ids.size(0), input_ids.size(1)), 
                                     dtype=torch.bool, device=input_ids.device)
        
        # Create causal mask
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), 
                                          dtype=torch.bool, device=input_ids.device),
                               diagonal=1)
        
        # Combine attention mask with causal mask
        attention_mask = attention_mask.bool()  # Ensure boolean type
        attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq]
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token = top_k_indices[0][torch.multinomial(F.softmax(top_k_logits, dim=-1)[0], 1)]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
                    
        return input_ids 