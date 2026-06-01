# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for fuzzy prefix matching."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FuzzyMatchConfig:
    """Fuzzy matching config. All fuzzy matching is opt-in."""
    
    # Enable fuzzy prefix matching
    enable_fuzzy_match: bool = False
    
    # Minimum token span a provider may reuse. Partial exact anchors shorter
    # than this are skipped; zero exact-prefix matches remain eligible.
    fuzzy_min_match_length: int = 16
    
    # Cosine-similarity threshold for SemanticEmbedding matches.
    # Range [0.0, 1.0].
    # Higher = stricter (fewer matches, higher precision); lower =
    # more permissive. Below ~0.50 alignment quality drops quickly.
    fuzzy_semantic_threshold: float = 0.60
    
    # Provider class for fuzzy matching logic.
    fuzzy_match_provider: str = "SemanticEmbedding"
    
    # Cache fuzzy match results for future reuse
    cache_fuzzy_results: bool = True
    
    # Maximum donors kept by the semantic provider.
    semantic_max_entries: int = 10000
    
    # Provider-internal donor chunk size.
    fuzzy_block_size: int = 16
    
    # Embedding model name for SemanticEmbeddingProvider
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # SemanticEmbedding-only: model architecture tag for SemBlend presets.
    model_arch: Optional[str] = None

    # SemanticEmbedding-only: minimum covered prompt fraction for a hit.
    fuzzy_min_reuse_ratio: float = 0.50

    def __post_init__(self):
        """Validate configuration values."""
        if self.fuzzy_min_match_length < 1:
            raise ValueError(
                f"fuzzy_min_match_length must be >= 1, got {self.fuzzy_min_match_length}"
            )
        
        if not (0.0 <= self.fuzzy_semantic_threshold <= 1.0):
            raise ValueError(
                f"fuzzy_semantic_threshold must be in [0.0, 1.0], "
                f"got {self.fuzzy_semantic_threshold}"
            )
        
        if self.fuzzy_match_provider not in ("SemanticEmbedding",):
            raise ValueError(
                f"fuzzy_match_provider must be 'SemanticEmbedding', "
                f"got {self.fuzzy_match_provider}"
            )
        
        if self.fuzzy_block_size < 1:
            raise ValueError(
                f"fuzzy_block_size must be >= 1, got {self.fuzzy_block_size}"
            )
        
        if self.semantic_max_entries < 1:
            raise ValueError(
                f"semantic_max_entries must be >= 1, "
                f"got {self.semantic_max_entries}"
            )

        if not (0.0 < self.fuzzy_min_reuse_ratio <= 1.0):
            raise ValueError(
                f"fuzzy_min_reuse_ratio must be in (0.0, 1.0], "
                f"got {self.fuzzy_min_reuse_ratio}"
            )

    @classmethod
    def from_server_args(cls, server_args):
        """Create FuzzyMatchConfig from ServerArgs."""
        return cls(
            enable_fuzzy_match=getattr(server_args, 'enable_fuzzy_match', False),
            fuzzy_min_match_length=getattr(server_args, 'fuzzy_min_match_length', 16),
            fuzzy_semantic_threshold=getattr(server_args, 'fuzzy_semantic_threshold', 0.60),
            fuzzy_match_provider=getattr(server_args, 'fuzzy_match_provider', 'SemanticEmbedding'),
            cache_fuzzy_results=getattr(server_args, 'cache_fuzzy_results', True),
            semantic_max_entries=getattr(server_args, 'semantic_max_entries', 10000),
            fuzzy_block_size=getattr(server_args, 'fuzzy_block_size', 16),
            embedding_model_name=getattr(server_args, 'embedding_model_name', 'all-MiniLM-L6-v2'),
            model_arch=getattr(server_args, 'fuzzy_model_arch', None),
            fuzzy_min_reuse_ratio=getattr(server_args, 'fuzzy_min_reuse_ratio', 0.50),
        )
