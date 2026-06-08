#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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
# This file is a part of the vllm-ascend project.
#

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d, nd_to_nz_spec

# NZ-format tile size; mask key dimensions are rounded up to a multiple of this.
_NZ_TILE = 16


def _round_up_to_tile(n: int, tile: int = _NZ_TILE) -> int:
    """Round ``n`` up to a multiple of ``tile`` (at least one tile)."""
    return max(((n + tile - 1) // tile) * tile, tile)


class AttentionMaskBuilder310:
    max_seqlen = 16384

    def __init__(self, device: torch.device, max_seqlen: int):
        """
        Initializes the AttentionMaskBuilder for the 310P device.

        Args:
            device (torch.device): The device on which tensors will be allocated.
            max_seqlen (int): Maximum length of a sequence (including prompt and generated text).
        """
        AttentionMaskBuilder310.max_seqlen = max_seqlen
        self.causal_attn_mask_cache = None
        self._cached_causal_len = 0
        self.non_causal_attn_mask_cache = None
        self._cached_non_causal_len = 0
        self.device = device

    @staticmethod
    def _build_splitfuse_additive_mask(position: torch.Tensor, key_len: int, device: torch.device) -> torch.Tensor:
        """Build the [num_query_tokens, key_len] causal additive mask directly (ND, before NZ cast).

        Row ``i`` is a query token at absolute position ``position[i]`` and attends to keys
        ``[0, position[i]]``: 0 where ``col <= position[i]``, -inf where ``col > position[i]``.
        This is identical to selecting row ``position[i]`` (sliced to ``key_len`` columns) from the
        full lower-triangular causal mask, but allocates only O(num_query_tokens * key_len) instead
        of the global O(max_seqlen^2) tensor.
        """
        col = torch.arange(key_len, device=device, dtype=torch.int32).unsqueeze(0)
        future = col > position.unsqueeze(1)
        mask = torch.zeros((position.shape[0], key_len), dtype=torch.float16, device=device)
        mask.masked_fill_(future, float("-inf"))
        return mask

    @staticmethod
    def gen_causal_additive_mask(max_seq_len: int, device: torch.device):
        """
        Generates a standard causal lower-triangular attention mask.

        The upper triangular part is filled with negative infinity (float("-inf"))
        to mask out future tokens, while the lower triangular part is kept as 0.

        Args:
            max_seq_len (int): The maximum sequence length for the mask.
            device (torch.device): The target device for the tensor.

        Returns:
            torch.Tensor: A float16 tensor representing the causal mask.
        """
        tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()
        upper = ~tril
        mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
        mask.masked_fill_(upper, float("-inf"))
        return mask

    @classmethod
    def get_splitfuse_mask(cls, attn_metadata: AscendMetadata, device: torch.device):
        """Build and format the SplitFuse attention mask for chunked prefill.

        The mask is built directly as ``[num_query_tokens, key_len]``, where ``key_len`` is the
        actual max context of the batch rounded up to the NZ tile size, then cast to FRACTAL_NZ.

        Args:
            attn_metadata (AscendMetadata): Metadata containing query start locations and sequence lengths.
            device (torch.device): The device to perform operations on.

        Returns:
            torch.Tensor: The splitfuse attention mask cast to ACL_FORMAT_FRACTAL_NZ.
        """
        qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
        qlens = qsl[1:] - qsl[:-1]
        q_list = qlens.tolist()
        context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
        c_list = context_lens.tolist()
        pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
        position = torch.tensor(pos_list, dtype=torch.int32, device=device)
        # Build the compact [num_query_tokens, key_len] mask directly instead of materializing
        # [max_seqlen, max_seqlen] and index-selecting rows (which OOMs the NPU at long context).
        max_ctx = int(context_lens.max().item()) if context_lens.numel() else 0
        key_len = _round_up_to_tile(max_ctx)
        splitfuse_mask = cls._build_splitfuse_additive_mask(position, key_len, device)
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(splitfuse_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    def get_attention_mask(self, causal: bool, model_config, actual_max_seqlen: int | None = None) -> torch.Tensor:
        """
        Retrieves the dense causal attention mask for the current batch.

        Args:
            causal (bool): Whether to generate a causal mask.
            model_config: Configuration object containing runner details.
            actual_max_seqlen (int | None): Actual max sequence length of the current batch.
                When provided, the mask is sized to this length (rounded up to the NZ tile)
                instead of the static ``max_seqlen``. This avoids allocating an
                O(max_model_len^2) mask, which OOMs the NPU at long context (e.g. 256K -> ~34 GiB fp16).
                Falls back to ``self.max_seqlen`` when not provided (back-compat).

        Returns:
            torch.Tensor: The attention mask in ACL_FORMAT_FRACTAL_NZ.

        Note:
            This dense mask is only consumed by the ``PrefillNoCache`` attention state. It is still
            O(actual_max_seqlen^2): a single un-chunked prefill of a very long prompt can OOM here.
            Long context MUST go through chunked prefill (``ChunkedPrefill`` -> ``get_splitfuse_mask``,
            which is O(num_query_tokens * actual_ctx)); ``DecodeOnly`` needs no mask. Disabling chunked
            prefill re-exposes the O(actual_len^2) allocation.
        """
        seqlen = self.max_seqlen if actual_max_seqlen is None else _round_up_to_tile(actual_max_seqlen)

        if getattr(model_config, "runner_type", None) == "pooling":
            if causal:
                return self._get_causal_mask(seqlen)
            else:
                return self._get_non_causal_mask(seqlen, model_config.dtype)

        return self._get_causal_mask(seqlen)

    def _get_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Internal method to get or update the cached causal attention mask.

        If the cache is empty or the requested length exceeds the cached length,
        a new mask is generated and converted to the NPU fractal format. The cache
        grows monotonically: it keeps the largest mask seen and is not shrunk, so a
        short request after a long one still holds the large allocation.

        Args:
            max_seq_len (int): The required sequence length.

        Returns:
            torch.Tensor: The cached causal mask in ACL_FORMAT_FRACTAL_NZ.
        """
        if self.causal_attn_mask_cache is None or self._cached_causal_len < max_seq_len:
            attn_mask = self.gen_causal_additive_mask(max_seq_len, self.device)
            self.causal_attn_mask_cache = torch_npu.npu_format_cast(nd_to_nz_2d(attn_mask), ACL_FORMAT_FRACTAL_NZ)
            self._cached_causal_len = max_seq_len
        return self.causal_attn_mask_cache

    def _get_non_causal_mask(self, max_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Internal method to get or update the cached non-causal attention mask.

        If the cache is empty or the requested length exceeds the cached length,
        a new mask is generated and converted to the NPU fractal format.

        Args:
            max_seq_len (int): The required sequence length.

        Returns:
            torch.Tensor: The cached non-causal mask in ACL_FORMAT_FRACTAL_NZ.
        """
        if self.non_causal_attn_mask_cache is None or self._cached_non_causal_len < max_seq_len:
            attention_mask_npu = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype, device=self.device)
            attention_mask_npu = nd_to_nz_2d(attention_mask_npu)
            self.non_causal_attn_mask_cache = torch_npu.npu_format_cast(
                attention_mask_npu.contiguous(), ACL_FORMAT_FRACTAL_NZ
            )
            self._cached_non_causal_len = max_seq_len

        return self.non_causal_attn_mask_cache
