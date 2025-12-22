"""
Transformer Model Architectures - Implemented from Scratch

This module contains three transformer variants as required by the project:

1. TextClassifier (Encoder-Only): 
   - Classifies paper abstracts into topics (NLP/CV/Security/Healthcare)
   - Uses bidirectional self-attention
   - Mean pooling over sequence for classification

2. DecoderOnlyLM (Decoder-Only):
   - Tiny language model for next-token prediction
   - Uses causal (masked) self-attention
   - Trained on paper abstracts corpus

3. EncoderDecoderTransformer (Encoder-Decoder):
   - Generates TL;DR summaries from abstracts
   - Encoder processes input, decoder generates output
   - Uses cross-attention between encoder and decoder

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Annotated Transformer (Harvard NLP)
- minGPT (Andrej Karpathy)

Model Constraints (as per project requirements):
- d_model: 128-256
- layers: 2-4  
- heads: 2-4
- max_seq_len: 128-256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .layers import (
    PositionalEncoding, 
    MultiHeadAttention, 
    FeedForward, 
    create_causal_mask,
    create_padding_mask
)


# =============================================================================
# ENCODER BLOCK
# =============================================================================

class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Structure:
    x -> MultiHeadAttention -> Add & Norm -> FFN -> Add & Norm -> output
    
    Uses pre-norm (LayerNorm before sublayer) for better training stability.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int = 256, 
        num_heads: int = 4,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Sub-layers
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional padding mask
        
        Returns:
            [batch, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


# =============================================================================
# DECODER BLOCK
# =============================================================================

class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Structure (for encoder-decoder):
    x -> Masked Self-Attention -> Add & Norm 
      -> Cross-Attention (with encoder) -> Add & Norm 
      -> FFN -> Add & Norm -> output
    
    Structure (for decoder-only LM):
    x -> Masked Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        is_decoder_only: If True, skip cross-attention (for GPT-style models)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = None,
        dropout: float = 0.1,
        is_decoder_only: bool = False
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        self.is_decoder_only = is_decoder_only
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (only for encoder-decoder models)
        if not is_decoder_only:
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [batch, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch, src_seq_len, d_model] (optional)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Padding mask for cross-attention
        
        Returns:
            [batch, tgt_seq_len, d_model]
        """
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention with encoder output
        if not self.is_decoder_only and encoder_output is not None:
            cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
            x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


# =============================================================================
# MODEL 1: ENCODER-ONLY (Text Classifier)
# =============================================================================

class TextClassifier(nn.Module):
    """
    Encoder-Only Transformer for Text Classification.
    
    Task: Classify paper abstracts into topics (NLP/CV/Security/Healthcare/ML)
    
    Architecture:
    - Token Embedding + Positional Encoding
    - N Encoder Blocks with bidirectional self-attention
    - Mean pooling over sequence
    - Classification head
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (128-256)
        num_heads: Number of attention heads (2-4)
        num_layers: Number of encoder blocks (2-4)
        num_classes: Number of classification categories
        max_seq_len: Maximum sequence length (128-256)
        dropout: Dropout rate
        pad_token_id: Padding token ID for masking
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 5,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
        
        Returns:
            Logits [batch, num_classes]
        """
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Convert to attention mask format [batch, 1, 1, seq_len]
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Embed tokens
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Mean pooling (excluding padding)
        mask_expanded = attention_mask.unsqueeze(-1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


# =============================================================================
# MODEL 2: DECODER-ONLY (Language Model)
# =============================================================================

class DecoderOnlyLM(nn.Module):
    """
    Decoder-Only Transformer Language Model (GPT-style).
    
    Task: Next-token prediction on paper abstracts (tiny LM)
    
    Architecture:
    - Token Embedding + Positional Encoding
    - N Decoder Blocks with causal self-attention
    - Language model head (projects to vocabulary)
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (128-256)
        num_heads: Number of attention heads (2-4)
        num_layers: Number of decoder blocks (2-4)
        max_seq_len: Maximum sequence length (128-256)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Decoder stack (with causal masking, no cross-attention)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout=dropout, is_decoder_only=True)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Language model head (project to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Target labels for loss computation [batch, seq_len]
        
        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len, device)
        
        # Embed tokens
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, self_attn_mask=causal_mask)
        
        # Final norm and project to vocabulary
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so we predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens
            eos_token_id: Stop generation at this token
        
        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Truncate if too long
            input_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self.forward(input_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids


# =============================================================================
# MODEL 3: ENCODER-DECODER (Summarization)
# =============================================================================

class EncoderDecoderTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for Sequence-to-Sequence tasks.
    
    Task: Generate TL;DR summaries from paper abstracts
    
    Architecture:
    - Encoder: Processes source sequence (abstract)
    - Decoder: Generates target sequence (summary) with cross-attention
    
    Args:
        vocab_size: Vocabulary size (shared between encoder/decoder)
        d_model: Model dimension (128-256)
        num_heads: Number of attention heads (2-4)
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        max_seq_len: Maximum sequence length (128-256)
        dropout: Dropout rate
        pad_token_id: Padding token ID
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Shared embedding for encoder and decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout=dropout, is_decoder_only=False)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src_ids: Source token IDs [batch, src_len]
            src_mask: Source padding mask
        
        Returns:
            Encoder output [batch, src_len, d_model]
        """
        # Create mask if not provided
        if src_mask is None:
            src_mask = (src_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2).float()
        
        # Embed
        x = self.embedding(src_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        
        # Encode
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence with encoder context.
        
        Args:
            tgt_ids: Target token IDs [batch, tgt_len]
            encoder_output: Encoder output [batch, src_len, d_model]
            tgt_mask: Target causal + padding mask
            memory_mask: Cross-attention mask
        
        Returns:
            Decoder output [batch, tgt_len, d_model]
        """
        batch_size, tgt_len = tgt_ids.shape
        device = tgt_ids.device
        
        # Create causal mask for decoder
        causal_mask = create_causal_mask(tgt_len, device)
        
        # Embed
        x = self.embedding(tgt_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        
        # Decode with cross-attention
        for layer in self.decoder_layers:
            x = layer(
                x, 
                encoder_output,
                self_attn_mask=causal_mask,
                cross_attn_mask=memory_mask
            )
        
        return self.decoder_norm(x)
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            src_ids: Source sequence [batch, src_len]
            tgt_ids: Target sequence (input) [batch, tgt_len]
            labels: Target labels for loss [batch, tgt_len]
        
        Returns:
            Tuple of (logits, loss)
        """
        # Encode source
        encoder_output = self.encode(src_ids)
        
        # Create memory mask for cross-attention
        memory_mask = (src_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2).float()
        
        # Decode
        decoder_output = self.decode(tgt_ids, encoder_output, memory_mask=memory_mask)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_new_tokens: int = 50,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate summary autoregressively.
        
        Args:
            src_ids: Source sequence [batch, src_len]
            max_new_tokens: Maximum tokens to generate
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            temperature: Sampling temperature
        
        Returns:
            Generated sequence [batch, gen_len]
        """
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        # Encode source once
        encoder_output = self.encode(src_ids)
        memory_mask = (src_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2).float()
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            # Decode current sequence
            decoder_output = self.decode(generated, encoder_output, memory_mask=memory_mask)
            
            # Get logits for last position
            logits = self.output_proj(decoder_output[:, -1, :]) / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, name: str = "Model") -> str:
    """Get a summary string for a model."""
    total_params = count_parameters(model)
    return f"{name}: {total_params:,} trainable parameters"


# Quick test
if __name__ == "__main__":
    print("Testing Transformer Models...")
    
    # Test parameters
    vocab_size = 10000
    d_model = 128
    num_heads = 4
    num_layers = 2
    batch_size = 2
    seq_len = 32
    
    # Test TextClassifier
    print("\n1. Testing TextClassifier (Encoder-Only)...")
    classifier = TextClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=5
    )
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = classifier(input_ids)
    print(f"   Input: {input_ids.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   {get_model_summary(classifier, 'TextClassifier')}")
    
    # Test DecoderOnlyLM
    print("\n2. Testing DecoderOnlyLM (Decoder-Only)...")
    lm = DecoderOnlyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )
    logits, loss = lm(input_ids, labels=input_ids)
    print(f"   Input: {input_ids.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   {get_model_summary(lm, 'DecoderOnlyLM')}")
    
    # Test generation
    generated = lm.generate(input_ids[:, :5], max_new_tokens=10)
    print(f"   Generated: {generated.shape}")
    
    # Test EncoderDecoderTransformer
    print("\n3. Testing EncoderDecoderTransformer...")
    seq2seq = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )
    src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(0, vocab_size, (batch_size, 16))
    logits, loss = seq2seq(src_ids, tgt_ids, labels=tgt_ids)
    print(f"   Source: {src_ids.shape}")
    print(f"   Target: {tgt_ids.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   {get_model_summary(seq2seq, 'EncoderDecoderTransformer')}")
    
    # Test generation
    generated = seq2seq.generate(src_ids, max_new_tokens=10)
    print(f"   Generated: {generated.shape}")
    
    print("\n🎉 All model tests passed!")
