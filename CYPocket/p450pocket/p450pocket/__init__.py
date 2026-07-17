"""Reproducible P450 ligand-pocket residue prediction.

The package root intentionally imports no numerical framework.  Structure
feature extraction runs in a legacy TensorFlow environment, whereas training
and classification use PyTorch; eager imports would couple those two stages.
"""

__all__ = []
__version__ = "0.1.0"
