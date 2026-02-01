"""
Transformer for next-token prediction on a custom finite vocabulary.
Uses Hugging Face GPT-2: custom vocab via config, no tokenizer, KV cache, SDPA/Flash Attention.
"""

from transformers import GPT2Config, GPT2LMHeadModel


def create_ntp_model(
    vocab_size: int,
    max_position_embeddings: int = 1024,
    n_embd: int = 128,
    n_layer: int = 6,
    n_head: int = 4,
    n_inner: int | None = None,
    attn_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    use_cache: bool = True,
    attn_implementation: str = "flash_attention_2",
) -> GPT2LMHeadModel:
    """
    Create a GPT-2 LM for next-token prediction with a custom finite vocabulary.

    No pretrained weights: embeddings are learned from scratch for vocab_size.
    Pass token IDs directly (no tokenizer). KV cache is used when use_cache=True.
    Uses Flash Attention 2 by default; requires the flash-attn package. Use
    attn_implementation="sdpa" to fall back to PyTorch SDPA.

    Returns:
        GPT2LMHeadModel with config for custom vocab_size and positional embeddings.
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_position_embeddings,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop,
        use_cache=use_cache,
        bos_token_id=0,
        eos_token_id=0,
        _attn_implementation=attn_implementation,
    )
    model = GPT2LMHeadModel.from_config(config)
    return model
