# Phase 1: Transformer Implementations from Scratch
# Contains: Encoder-only, Decoder-only, Encoder-Decoder architectures

# Lazy imports to avoid circular dependencies
__all__ = [
    'PositionalEncoding',
    'MultiHeadAttention', 
    'FeedForward',
    'EncoderBlock',
    'DecoderBlock',
    'TextClassifier',
    'DecoderOnlyLM',
    'EncoderDecoderTransformer'
]

def __getattr__(name):
    """Lazy import for better performance and avoiding circular imports."""
    if name in ['PositionalEncoding', 'MultiHeadAttention', 'FeedForward']:
        from .layers import PositionalEncoding, MultiHeadAttention, FeedForward
        return locals()[name]
    elif name in ['EncoderBlock', 'DecoderBlock', 'TextClassifier', 'DecoderOnlyLM', 'EncoderDecoderTransformer']:
        from .models import EncoderBlock, DecoderBlock, TextClassifier, DecoderOnlyLM, EncoderDecoderTransformer
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
