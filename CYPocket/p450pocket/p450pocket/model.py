"""BERT residue classifier compatible with the supplied legacy weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from transformers import BertConfig, BertModel


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings required to reconstruct a checkpoint."""

    feature_dim: int = 100
    num_labels: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    chunk_size: int = 512
    architecture: str = "bert-base-residue-classifier"
    compatibility_mode: str = "legacy-v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "ModelConfig":
        if not values:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def bert_config(self) -> BertConfig:
        # Explicit values avoid silently changing the architecture when the
        # transformers package changes its defaults.
        return BertConfig(
            vocab_size=30522,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
        )


class ResidueBertClassifier(nn.Module):
    """Classify each residue from a 100-dimensional structural embedding.

    Compatibility notes
    -------------------
    The historical class declared a separate dropout layer but never applied
    it in ``forward``. This implementation deliberately preserves that
    behavior. BERT's own 0.1 dropout remains active during training.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.model_config = config or ModelConfig()
        self.bert = BertModel(
            self.model_config.bert_config(), add_pooling_layer=True
        )
        self.feature_projection = nn.Linear(
            self.model_config.feature_dim, self.model_config.hidden_size
        )
        self.classifier = nn.Linear(
            self.model_config.hidden_size, self.model_config.num_labels
        )

    @classmethod
    def from_pretrained_bert(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[ModelConfig] = None,
        *,
        local_files_only: bool = True,
    ) -> "ResidueBertClassifier":
        """Initialize BERT from a local pretrained model for de novo training.

        The projection and residue classifier retain their deterministic random
        initialization.  Existing P450 checkpoints should instead be loaded
        with :func:`p450pocket.checkpoint.load_model`.
        """

        model = cls(config)
        model.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path,
            config=model.model_config.bert_config(),
            add_pooling_layer=True,
            local_files_only=local_files_only,
        )
        return model

    def forward(self, features: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"features must have shape [batch, residues, {self.model_config.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.model_config.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.model_config.feature_dim}, got {features.shape[-1]}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                features.shape[:2], dtype=torch.long, device=features.device
            )
        if attention_mask.shape != features.shape[:2]:
            raise ValueError(
                f"attention_mask must have shape {tuple(features.shape[:2])}, "
                f"got {tuple(attention_mask.shape)}"
            )
        embedded = self.feature_projection(features)
        outputs = self.bert(inputs_embeds=embedded, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state)

    def forward_chunked(
        self,
        features: Tensor,
        attention_mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        """Run independent, non-overlapping windows as used for legacy training."""

        if features.ndim != 3:
            raise ValueError(f"features must be 3D, got {tuple(features.shape)}")
        length = int(features.shape[1])
        if length < 1:
            raise ValueError("a protein must contain at least one residue")
        size = int(chunk_size or self.model_config.chunk_size)
        if size < 1 or size > self.model_config.max_position_embeddings:
            raise ValueError(
                f"chunk_size must be in [1, {self.model_config.max_position_embeddings}], got {size}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                features.shape[:2], dtype=torch.long, device=features.device
            )
        chunks = []
        for start in range(0, length, size):
            stop = min(start + size, length)
            chunks.append(
                self.forward(
                    features[:, start:stop, :],
                    attention_mask[:, start:stop],
                )
            )
        return torch.cat(chunks, dim=1)


def positive_probabilities(logits: Tensor) -> Tensor:
    """Return P(pocket) for binary logits."""

    if logits.shape[-1] != 2:
        raise ValueError(f"binary logits must end in dimension 2, got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=-1)[..., 1]
