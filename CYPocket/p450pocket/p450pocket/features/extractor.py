"""High-level, residue-safe PocketMiner feature extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from ..structure import ResidueKey, StructureRecord, parse_feature_protein


class FeatureExtractionError(RuntimeError):
    """Raised when the TensorFlow backend or checkpoint cannot produce features."""


@dataclass(frozen=True)
class PocketMinerConfig:
    node_features: Tuple[int, int] = (8, 50)
    edge_features: Tuple[int, int] = (1, 32)
    hidden_dim: Tuple[int, int] = (16, 100)
    num_layers: int = 4
    k_neighbors: int = 30
    dropout: float = 0.1


@dataclass(frozen=True)
class FeatureResult:
    """Feature matrix and the exact residue identity represented by each row."""

    features: np.ndarray
    residue_keys: Tuple[ResidueKey, ...]
    checkpoint_prefix: Path
    checkpoint_sha256: str
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float32).copy()
        expected = (len(self.residue_keys), 100)
        if values.shape != expected:
            raise FeatureExtractionError(
                f"Feature shape {values.shape} does not match expected {expected}"
            )
        if not np.isfinite(values).all():
            raise FeatureExtractionError("PocketMiner produced NaN or infinite features")
        values.setflags(write=False)
        object.__setattr__(self, "features", values)


def _checkpoint_files(prefix: Path) -> Tuple[Path, ...]:
    index = Path(f"{prefix}.index")
    data_files = tuple(sorted(prefix.parent.glob(f"{prefix.name}.data-*")))
    if not index.is_file() or not data_files:
        raise FeatureExtractionError(
            "TensorFlow checkpoint is incomplete. Expected "
            f"{index} and at least one {prefix.name}.data-* file"
        )
    return (index,) + data_files


def _checkpoint_digest(files: Tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class PocketMinerExtractor:
    """Load the legacy TensorFlow encoder once and extract ``[L, 100]`` tokens.

    Chains are processed independently because the available classifier was
    trained on single-chain AlphaFold structures.  Combining chains in one GVP
    graph would introduce artificial sequential orientations at chain breaks.
    """

    def __init__(
        self,
        checkpoint_prefix: Path | str,
        *,
        config: PocketMinerConfig = PocketMinerConfig(),
    ) -> None:
        prefix = Path(checkpoint_prefix).expanduser()
        if prefix.suffix == ".index":
            prefix = prefix.with_suffix("")
        prefix = prefix.resolve()
        files = _checkpoint_files(prefix)

        self.checkpoint_prefix = prefix
        self.checkpoint_sha256 = _checkpoint_digest(files)
        self.config = config
        self._tf = None
        self._model = None
        self._checkpoint = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import tensorflow as tf

            from .pocketminer import MQAModel
        except Exception as exc:
            raise FeatureExtractionError(
                "PocketMiner requires a working TensorFlow installation compatible "
                f"with the legacy checkpoint; backend import failed: {exc}"
            ) from exc

        if not tf.executing_eagerly():
            raise FeatureExtractionError(
                "PocketMinerExtractor requires TensorFlow eager execution"
            )

        model = MQAModel(
            node_features=self.config.node_features,
            edge_features=self.config.edge_features,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            k_neighbors=self.config.k_neighbors,
            dropout=self.config.dropout,
        )

        # A finite, non-collinear dummy chain builds every feature-encoder
        # variable before strict checkpoint validation.  Sequence length does
        # not affect variable shapes.
        dummy = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.2, 0.2, 0.0]],
                [[0.3, 0.1, 0.0], [0.4, 0.1, 0.0], [0.5, 0.2, 0.0], [0.5, 0.3, 0.0]],
                [[0.6, 0.2, 0.0], [0.7, 0.2, 0.0], [0.8, 0.3, 0.0], [0.8, 0.4, 0.0]],
            ],
            dtype=np.float32,
        )[None, ...]
        dummy_sequence = np.zeros((1, 3), dtype=np.int32)
        dummy_mask = np.ones((1, 3), dtype=np.float32)
        output = model(dummy, dummy_sequence, dummy_mask, train=False, res_level=True)
        if tuple(output.shape) != (1, 3, 100):
            raise FeatureExtractionError(
                f"Unexpected PocketMiner build output shape: {tuple(output.shape)}"
            )

        # The historical saver also tracked an Adam optimizer, but restoring
        # only the named ``model`` subtree avoids version-specific optimizer
        # slot incompatibilities while still asserting every built model
        # variable below.
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(str(self.checkpoint_prefix))
        try:
            if hasattr(status, "assert_nontrivial_match"):
                status.assert_nontrivial_match()
            status.assert_existing_objects_matched()
        except (AssertionError, ValueError) as exc:
            # The historical TF 2.6 checkpoint serialised ``save_counter`` as
            # a trackable node without a VARIABLE_VALUE attribute.  A fresh
            # Checkpoint creates it as a scalar variable, so TensorFlow reports
            # that bookkeeping object as unmatched even when every model
            # variable is restored.  No other unmatched variable is accepted.
            unmatched = re.findall(r"<tf\.Variable '([^']+)'", str(exc))
            if unmatched != ["save_counter:0"]:
                raise FeatureExtractionError(
                    f"PocketMiner checkpoint does not match the legacy architecture: {exc}"
                ) from exc
        # Historical checkpoints can contain unused classification-head or
        # optimizer-slot entries.  Model variables above have already been
        # asserted, so acknowledging those extras is safe and intentional.
        status.expect_partial()

        self._tf = tf
        self._model = model
        self._checkpoint = checkpoint

    def extract(self, structure: StructureRecord) -> FeatureResult:
        if self._model is None or self._tf is None:
            raise FeatureExtractionError("PocketMiner backend is not loaded")

        features = np.empty((len(structure), 100), dtype=np.float32)
        warnings = list(structure.warnings)
        if len(structure.chains) > 1:
            warnings.append(
                f"Processed {len(structure.chains)} chains independently for "
                "single-chain model compatibility"
            )

        all_sequence_indices = structure.sequence_indices
        for chain_id in structure.chains:
            residues = structure.chain_residues(chain_id)
            if not residues:
                continue
            # Bio.PDB reports Å.  Dividing by ten exactly reproduces the nm
            # units returned by MDTraj in the historical extractor.
            coordinates_angstrom = np.stack(
                [residue.backbone_angstrom for residue in residues]
            )
            # Match MDTraj's PDB parsing/conversion order exactly; see
            # StructureRecord.backbone_coordinates_nm.
            coordinates_nm = np.asarray(
                np.round(coordinates_angstrom.astype(np.float64), decimals=3) / 10.0,
                dtype=np.float32,
            )[None, ...]
            sequence = np.asarray(
                [
                    # StructureRecord already validates the legacy alphabet.
                    all_sequence_indices[residue.row_index]
                    for residue in residues
                ],
                dtype=np.int32,
            )[None, ...]
            mask = np.ones((1, len(residues)), dtype=np.float32)
            try:
                output = self._model(
                    coordinates_nm,
                    sequence,
                    mask,
                    train=False,
                    res_level=True,
                )
                values = np.asarray(output.numpy(), dtype=np.float32)
            except Exception as exc:
                raise FeatureExtractionError(
                    f"PocketMiner failed for {structure.source_path}, chain "
                    f"{chain_id or '_'}: {exc}"
                ) from exc
            expected = (1, len(residues), 100)
            if values.shape != expected:
                raise FeatureExtractionError(
                    f"PocketMiner returned {values.shape} for chain {chain_id or '_'}, "
                    f"expected {expected}"
                )
            row_indices = [residue.row_index for residue in residues]
            features[row_indices, :] = values[0]

        return FeatureResult(
            features=features,
            residue_keys=structure.residue_keys,
            checkpoint_prefix=self.checkpoint_prefix,
            checkpoint_sha256=self.checkpoint_sha256,
            warnings=tuple(warnings),
        )

    def extract_pdb(
        self,
        path: Path | str,
        *,
        model_index: int = 0,
        chains: Optional[Tuple[str, ...]] = None,
        modified_residue_map: Optional[Mapping[str, str]] = None,
        reject_multiple_models: bool = True,
    ) -> FeatureResult:
        structure = parse_feature_protein(
            path,
            model_index=model_index,
            chains=chains,
            modified_residue_map=modified_residue_map,
            reject_multiple_models=reject_multiple_models,
        )
        return self.extract(structure)
