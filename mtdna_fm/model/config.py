"""
MtDNA foundation model configuration.

WHY inherit PretrainedConfig (not a plain dataclass):
  HuggingFace's PretrainedConfig provides:
    - save_pretrained() / from_pretrained(): serialises to config.json
      alongside model weights so the architecture is self-describing.
    - Automatic handling of model_type, which PEFT uses to determine
      which layer names to target for LoRA.
    - to_dict() / to_json_string() for experiment tracking (MLflow).
  Using it from the start avoids painful retrofitting later.

Novel fields vs standard BERT config:
  genome_length: 16569 — the circular mtDNA genome is exactly this length.
    Used by MtDNACircularPositionalEncoding to pre-compute the angular
    frequency for each position.
  use_circular_encoding: when True, position IDs are mapped through a
    circular sinusoidal PE buffer. When False, falls back to standard
    learnable positional embeddings (useful for ablations).
  use_het_projection: when True, a continuous heteroplasmy channel
    (float in [0, 1]) is projected into the embedding space alongside
    the k-mer token IDs.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class MtDNAConfig(PretrainedConfig):
    """
    Configuration for the mtDNA foundation model encoder.

    All hyperparameters documented with biological/technical rationale.
    """

    model_type = "mtdna_fm"

    def __init__(
        self,
        # Vocabulary
        vocab_size: int = 4102,  # 4096 6-mers + 6 special tokens
        # Transformer dimensions
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        # Sequence
        max_seq_len: int = 514,  # 512 k-mer tokens + CLS + SEP
        # mtDNA-specific: circular genome properties
        genome_length: int = 16569,
        use_circular_encoding: bool = True,
        use_het_projection: bool = True,
        # Regularisation
        dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        # Special token IDs — must match KmerVocabulary.SPECIAL_TOKENS
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        mask_token_id: int = 2,
        unk_token_id: int = 3,
        sep_token_id: int = 4,
        het_token_id: int = 5,
        **kwargs: object,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.genome_length = genome_length
        self.use_circular_encoding = use_circular_encoding
        self.use_het_projection = use_het_projection
        self.dropout_prob = dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.cls_token_id = cls_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.sep_token_id = sep_token_id
        self.het_token_id = het_token_id
