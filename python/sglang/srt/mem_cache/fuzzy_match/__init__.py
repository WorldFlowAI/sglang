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

"""Fuzzy prefix matching support for RadixCache."""

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    create_fuzzy_match_provider,
)
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    as_long_tensor,
    copy_kv_with_rope_correction,
)

__all__ = [
    "FuzzyMatchConfig",
    "FuzzyMatchProvider",
    "FuzzyMatchResult",
    "as_long_tensor",
    "copy_kv_with_rope_correction",
    "create_fuzzy_match_provider",
]
