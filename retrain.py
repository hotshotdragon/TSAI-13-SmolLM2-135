import os
import math
import time
import inspect
from click import Option
from numpy import dtype
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import sys
from typing import Optional, Tuple
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continue_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Continue Training Script Started")
logging.info("Script Started")

@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    vocab_size: int = 50257
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = True
    use_cache: bool = True


def save_model(model, optimizer, loss, output_dir='saved_models'):
    from datetime import datetime
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smollm2-135_{timestamp}.pt"
    filepath = os.path.join(output_dir, filename)
    
    save_dict = {
        'model_state_dict': model.state_dict(),  # Fixed key name
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'loss': loss
    }
    torch.save(save_dict, filepath)
    logging.info(f"Model Saved to {filepath}")
    return filepath

def apply_rotary_pos_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    sin, cos = rotary_emb[..., :head_dim // 2], rotary_emb[..., head_dim // 2:]
    rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated_x

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len <= 0:
            logging.warning(f"Invalid seq_len: {seq_len}, setting to 1")
            seq_len = 1
        if seq_len > self.max_position_embeddings:
            logging.warning(f"seq_len {seq_len} exceeds max_position_embeddings {self.max_position_embeddings}, truncating")
            seq_len = self.max_position_embeddings
            
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        sincos = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        # Change here: rearrange dimensions so that seq_len is in dimension 2
        return emb[None, None, :, :]

class LlamaMLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Initialize weights
        std = config.initializer_range
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(layer.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        std = config.initializer_range
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(layer.weight, std=std)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positional_ids: Optional[torch.Tensor] = None,
        pass_key_value: Optional[torch.Tensor] = None,  # Fix: Made parameter name consistent
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        if rotary_emb is not None:
            query_states = apply_rotary_pos_emb(query_states, rotary_emb)
            key_states = apply_rotary_pos_emb(key_states, rotary_emb)
            
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value_states)
        
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size, seq_length, self.hidden_size)
        hidden_states = self.o_proj(hidden_states)
        return hidden_states

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positional_ids: Optional[torch.Tensor] = None,
        pass_key_value: Optional[Tuple[torch.Tensor]] = None,  # Fixed parameter name
        rotary_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            pass_key_value=pass_key_value,  # Fixed parameter name
            rotary_emb=rotary_emb
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,  # Fixed: Use num_attention_heads instead of num_key_value_heads
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta)
        
        nn.init.normal_(self.embed_tokens.weight, std=config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positional_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        # Add debug logging
        logging.debug(f"Input shape: {input_ids.shape}, Hidden states shape: {hidden_states.shape}")
        logging.debug(f"Max position embeddings: {self.config.max_position_embeddings}")
        
        # Ensure seq_len is valid
        seq_len = hidden_states.shape[1]
        if seq_len > self.config.max_position_embeddings:
            seq_len = self.config.max_position_embeddings
            logging.warning(f"Sequence length {hidden_states.shape[1]} exceeds max_position_embeddings, truncating to {seq_len}")
        
        rotary_emb = self.rotary_emb(hidden_states, seq_len)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                positional_ids=positional_ids,
                rotary_emb=rotary_emb
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positional_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        
        return logits
# For this example, we'll assume the definitions of LlamaForCausalLM and SmolLM2Config are available in this file.

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # logits: shape (batch_size, vocab_size)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value
    
    return logits

def generate_text(model, prompt, max_new_tokens=50, temperature=1.2, top_k=50, top_p=0.95):
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            # Filter logits with top-k and/or top-p
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = enc.decode(input_ids[0].tolist())
    model.train()
    return generated_text


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Specify the checkpoint file path (update the filename as needed)
checkpoint_path = os.path.join("saved_models", "smollm2-135_20250218_133517.pt")
logging.info(f"Loading checkpoint from {checkpoint_path}")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

# Recreate model and optimizer
model = LlamaForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

logging.info("Checkpoint loaded successfully.")

# Setup your dataloader
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('Lord-of-the-Rings.txt', 'r', encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        logging.info(f'loaded {len(self.tokens)} tokens')
        logging.info(f'1 epoch = {len(self.tokens) // (B*T)} batches')
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

train_loader = DataLoaderLite(B=2, T=1024)

# Define training hyperparameters
current_step = 4999  # Assuming checkpoint was saved at step 5000
additional_steps = 51

def get_lr(it):
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 1000
    max_steps = 5000
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(math.pi * decay_ratio)))
    return float(min_lr + coeff * (max_lr - min_lr))

# Continue training for 50 additional steps
for step in range(current_step + 1, current_step + additional_steps + 1):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    loss = model(x, labels=y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optionally update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    
    logging.info(f'step {step} | loss: {loss.item():.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.2f}')
    
    # Optionally, generate sample text every 10 steps
    if step % 10 == 0:
        prompt = "Through Rohan over fen and field where the long grass grows"
        generated_text = generate_text(model, prompt)
        logging.info(f"Generated text at step {step}:\n{generated_text}")

# Save the checkpoint after additional training
new_checkpoint_path = save_model(model, optimizer, loss, output_dir='saved_models')
logging.info(f"Continued training complete. New checkpoint saved to {new_checkpoint_path}")
