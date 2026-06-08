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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310, _round_up_to_tile


class TestAttentionMaskBuilder310(TestBase):
    def setUp(self):
        self.max_seqlen = 4096
        self.attention_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), self.max_seqlen)

    @patch("torch_npu.npu_format_cast")
    def test_get_attention_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        model_config = MagicMock()
        attn_mask = self.attention_mask_builder.get_attention_mask(causal=True, model_config=model_config)
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, self.max_seqlen, 16))
        self.assertEqual(attn_mask[0][-1][0][-1], torch.tensor(float("-inf"), dtype=torch.float16))

    @patch("torch_npu.npu_format_cast")
    def test_get_splitfuse_attn_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 1, 5])
        attn_metadata.seq_lens = torch.tensor([7, 4])
        attn_mask = self.attention_mask_builder.get_splitfuse_mask(attn_metadata, torch.device("cpu"))
        # Key dim is sized to the actual max context (max(seq_lens)=7 -> 1 NZ tile), not max_seqlen.
        self.assertEqual(attn_mask.shape, (1, _round_up_to_tile(7) // 16, 16, 16))

    def test_round_up_to_tile(self):
        self.assertEqual(_round_up_to_tile(0), 16)
        self.assertEqual(_round_up_to_tile(1), 16)
        self.assertEqual(_round_up_to_tile(15), 16)
        self.assertEqual(_round_up_to_tile(16), 16)
        self.assertEqual(_round_up_to_tile(17), 32)

    def test_splitfuse_additive_mask_matches_index_select(self):
        # The direct [num_query_tokens, key_len] build must equal selecting rows `position`
        # from the full causal mask (sliced to key_len) — same semantics, no O(max_seqlen^2) alloc.
        device = torch.device("cpu")
        n = 32
        full = AttentionMaskBuilder310.gen_causal_additive_mask(n, device)
        position = torch.tensor([0, 3, 7, 20, 31], dtype=torch.int32)
        for key_len in (16, 32):
            direct = AttentionMaskBuilder310._build_splitfuse_additive_mask(position, key_len, device)
            expected = full.index_select(0, position.to(torch.int64))[:, :key_len]
            self.assertEqual(direct.shape, (position.shape[0], key_len))
            torch.testing.assert_close(direct, expected)
