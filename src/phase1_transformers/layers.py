"""
Transformer Building Blocks - Implemented from Scratch

This module contains the fundamental components of the Transformer architecture:
1. Positional Encoding - Sinusoidal position embeddings
2. Multi-Head Attention - Scaled dot-product attention with multiple heads
3. Feed-Forward Network - Position-wise fully connected layers
4. Layer Normalization - For training stability

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Annotated Transformer (Harvard NLP)
- minGPT (Andrej Karpathy)

Implementation Constraints (as per project requirements):
- d_model: 128-256
- layers: 2-4
- heads: 2-4
- max_seq_len: 128-256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to learn relative positions because for any fixed offset k,
    PE(pos+k) can be represented as a linear function of PE(pos).
    
    Args:
        d_model: Dimension of the model (embedding size)
        max_len: Maximum sequence length to precompute
        dropout: Dropout rate applied after adding positional encoding
    """
    
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Position indices [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term: 10000^(2i/d_model)
        # Using exp(log) for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need".
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Store attention weights for visualization (optional)
        self.attention_weights = None
    
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: [batch, heads, seq_len, d_k]
            key: [batch, heads, seq_len, d_k]
            value: [batch, heads, seq_len, d_k]
            mask: Optional mask tensor
        
        Returns:
            Output tensor and attention weights
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # [batch, heads, seq_len, d_k] @ [batch, heads, d_k, seq_len]
        # -> [batch, heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Mask should be broadcastable to [batch, heads, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, d_k]
        # -> [batch, heads, seq_len, d_k]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        For self-attention: query = key = value
        For cross-attention: query from decoder, key/value from encoder
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # 1. Linear projections and reshape for multi-head
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        # -> [batch, num_heads, seq_len, d_k]
        
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        attn_output, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
        # -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. Final linear projection
        output = self.out_linear(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
    
    Typically d_ff = 4 * d_model in the original paper,
    but we use smaller values for efficiency.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden layer dimension (default: 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            [batch, seq_len, d_model]
        """
        # First linear + ReLU
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Second linear
        x = self.linear2(x)
        
        return x


class LayerNorm(nn.Module):
    """
    Layer Normalization (manual implementation for learning purposes).
    
    LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
    
    Note: PyTorch's nn.LayerNorm is more optimized. This is for educational purposes.
    
    Args:
        d_model: Normalization dimension
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            Normalized tensor [batch, seq_len, d_model]
        """
        # Compute mean and std along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize and scale
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (look-ahead) mask for decoder self-attention.
    
    Prevents attending to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        Causal mask [1, 1, seq_len, seq_len]
    """
    # Lower triangular matrix (True = attend, False = mask)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Add batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create padding mask to ignore pad tokens.
    
    Args:
        input_ids: Token IDs [batch, seq_len]
        pad_token_id: ID of padding token
    
    Returns:
        Padding mask [batch, 1, 1, seq_len]
    """
    # 1 where not padding, 0 where padding
    mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask.float()


# Quick test
if __name__ == "__main__":
    print("Testing Transformer Layers...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test PositionalEncoding
    pe = PositionalEncoding(d_model)
    out = pe(x)
    print(f"✅ PositionalEncoding: {x.shape} -> {out.shape}")
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model, num_heads)
    out = mha(x, x, x)
    print(f"✅ MultiHeadAttention: {x.shape} -> {out.shape}")
    
    # Test with causal mask
    mask = create_causal_mask(seq_len)
    out = mha(x, x, x, mask)
    print(f"✅ MultiHeadAttention (causal): {x.shape} -> {out.shape}")
    
    # Test FeedForward
    ff = FeedForward(d_model)
    out = ff(x)
    print(f"✅ FeedForward: {x.shape} -> {out.shape}")
    
    # Test LayerNorm
    ln = LayerNorm(d_model)
    out = ln(x)
    print(f"✅ LayerNorm: {x.shape} -> {out.shape}")
    
    print("\n🎉 All layer tests passed!")
