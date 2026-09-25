"""GPU-accelerated semantic embeddings using frozen all-MiniLM-L6-v2.

Owner: Neural Embeddings Module.
Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0, ~22M parameters << 8B limit).
Precomputes and caches dense vectors for names and addresses to calculate
semantic cosine similarity features for candidate pairs.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._name_embeddings: Dict[str, np.ndarray] = {}
        self._addr_embeddings: Dict[str, np.ndarray] = {}

    def load_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            print(f"Loading {self.model_name} on device: {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def compute_and_cache(
        self,
        entity_ids: List[str],
        names: List[str],
        addresses: List[str],
        batch_size: int = 256,
    ):
        """Batch encode names and addresses on GPU, storing normalized unit vectors."""
        self.load_model()
        print(f"Encoding {len(names)} names and addresses on {self.device} (batch_size={batch_size})...")

        # Encode names
        cleaned_names = [n if n else "" for n in names]
        name_vecs = self.model.encode(
            cleaned_names,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Encode addresses
        cleaned_addrs = [a if a else "" for a in addresses]
        addr_vecs = self.model.encode(
            cleaned_addrs,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for eid, nv, av in zip(entity_ids, name_vecs, addr_vecs):
            self._name_embeddings[eid] = nv
            self._addr_embeddings[eid] = av

    def get_cosine_similarities(
        self,
        s1_id: str,
        cand_id: str,
    ) -> Tuple[float, float]:
        """Compute cosine similarity between normalized vectors (dot product)."""
        name_sim = 0.0
        addr_sim = 0.0

        if s1_id in self._name_embeddings and cand_id in self._name_embeddings:
            nv1 = self._name_embeddings[s1_id]
            nv2 = self._name_embeddings[cand_id]
            name_sim = float(np.dot(nv1, nv2))

        if s1_id in self._addr_embeddings and cand_id in self._addr_embeddings:
            av1 = self._addr_embeddings[s1_id]
            av2 = self._addr_embeddings[cand_id]
            addr_sim = float(np.dot(av1, av2))

        return max(0.0, min(1.0, name_sim)), max(0.0, min(1.0, addr_sim))
